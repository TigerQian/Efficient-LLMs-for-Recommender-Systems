import json
import subprocess
import csv
import os
import time
import numpy as np
from collections import defaultdict
import re
from collections import Counter
import argparse
from openai import OpenAI
from dotenv import load_dotenv
import pathlib

# 加载.env文件中的环境变量
dotenv_path = pathlib.Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path)

# 初始化OpenAI客户端
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")
)

def personalized_generation(
        model: str = "gpt-4o",
        temperature: float = 0.1,
        top_p: float = 0.95,
        ctx: int = 13000,
        message: list = None,
        is_stream: bool = False
):
    """
    利用Ollama框架或OpenAI API将大模型封装成类似ChatGPT接口规范的API服务，基于提供的信息为你生产个性化的回答
    :param model: 模型名称，如"llama3.2"或"gpt-4o"
    :param temperature: 温度
    :param top_p: top_p
    :param ctx: 生成token长度
    :param message: 提供给大模型的对话信息
    :param is_stream: 是否流式输出大模型的应答
    :return: 基于prompt，利用大模型生成的结果
    """
    if message is None:
        message = [{}]
    
    # 判断是使用Ollama还是OpenAI
    if model.startswith("gpt") or model.startswith("o1-"):
        print(f"使用OpenAI API调用{model}模型")
        return _call_openai_api(model, temperature, top_p, message, is_stream)
    else:
        print(f"使用Ollama API调用{model}模型")
        return _call_ollama_api(model, temperature, top_p, ctx, message, is_stream)

def _call_openai_api(model, temperature, top_p, message, is_stream):
    """
    调用OpenAI API
    
    支持的模型类型包括：
    - gpt-4o: GPT-4o模型
    - o1-preview: OpenAI的o1-preview模型
    - o1-mini: OpenAI的o1-mini模型
    - 其他OpenAI支持的模型
    """
    try:
        # 处理o1系列模型不支持system角色的问题
        if model.startswith("o1-"):
            # 将system消息转换为user消息
            processed_messages = []
            for msg in message:
                if msg.get("role") == "system":
                    # 将system消息转换为user消息
                    processed_messages.append({
                        "role": "user",
                        "content": f"Instructions: {msg.get('content')}"
                    })
                else:
                    processed_messages.append(msg)
            message = processed_messages
            
            # o1系列模型不支持自定义temperature和top_p参数
            response = client.chat.completions.create(
                model=model,
                messages=message,
                stream=is_stream
            )
        else:
            # 其他模型使用完整参数
            response = client.chat.completions.create(
                model=model,
                messages=message,
                temperature=temperature,
                top_p=top_p,
                stream=is_stream
            )
        
        if is_stream:
            # 处理流式响应
            res = ""
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    content = chunk.choices[0].delta.content
                    if content:
                        res += content
                        print(content, end='', flush=True)
            print()  # 添加换行符
            return res
        else:
            # 处理非流式响应
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            return ""
    except Exception as e:
        print(f"OpenAI API调用错误: {str(e)}")
        return ""

def _call_ollama_api(model, temperature, top_p, ctx, message, is_stream):
    """
    调用Ollama API
    """
    # 构建API请求数据
    data = {
        "model": model,
        "messages": message,
        "stream": is_stream
    }
    
    # 使用subprocess调用curl命令
    cmd = [
        'curl', '-s', 'http://localhost:11434/api/chat',
        '-d', json.dumps(data)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return ""
    
    if is_stream:
        # 处理流式响应
        res = ""
        lines = result.stdout.strip().split('\n')
        for line in lines:
            try:
                data = json.loads(line)
                if 'message' in data and 'content' in data['message']:
                    content = data['message']['content']
                    res += content
                    print(content, end='', flush=True)
            except json.JSONDecodeError:
                pass
        print()  # 添加换行符
        return res
    else:
        # 处理非流式响应
        lines = result.stdout.strip().split('\n')
        for line in lines:
            try:
                data = json.loads(line)
                if 'message' in data and 'content' in data['message'] and data.get('done', False):
                    return data['message']['content']
            except json.JSONDecodeError:
                pass
        return ""


def single_step_rec(
        user_id: str,
        history_news_titles: list[str],
        candidate_news_with_ids: list[tuple[str, str]], # [(original_id, title), ...]
        model: str = "gpt-4o",
        temperature: float = 0.1,
        top_p: float = 0.95,
        ctx: int = 13000, # Retained for Ollama compatibility
        is_stream: bool = True,
        num_recommendations: int = 3
):
    """
    Generates news recommendations using the CHAT-RAC framework with a single LLM call.
    :param user_id: User ID for personalization.
    :param history_news_titles: List of user's browsed news titles.
    :param candidate_news_with_ids: List of candidate news, each as a tuple (original_id, title).
    :param model: Name of the large language model to use.
    :param temperature: Temperature parameter for generation.
    :param top_p: Top_p parameter for generation.
    :param ctx: Context length (primarily for Ollama).
    :param is_stream: Whether to use stream output.
    :param num_recommendations: Number of news articles to recommend.
    :return: Dictionary containing the LLM's raw output.
    """
    history_str = "- " + "\n- ".join(history_news_titles) if history_news_titles else "No browsing history provided."

    formatted_candidate_news = []
    for i, (original_id, title) in enumerate(candidate_news_with_ids, 1):
        # Format: 1. [N12345] "News Title"
        formatted_candidate_news.append(f"{i}. [{original_id}] \"{title}\"") # Escaped quotes
    candidate_str = "\n".join(formatted_candidate_news) if formatted_candidate_news else "No candidate news available."

    system_prompt = f'''### ROLE ###
You are an expert AI News Recommendation Assistant. Your primary goal is to identify user interests and provide relevant news recommendations with explanations.

### CONTEXT ###
You will be given:
1.  A User's Browsing History.
2.  A list of Candidate News articles with their original IDs.

### TASK & INSTRUCTIONS ###
Based *solely* on the provided User's Browsing History and the Candidate News:
1.  **Infer User Interests**: Identify the user's main topics and preferences from their browsing history.
2.  **Recommend Articles**: From the CANDIDATE NEWS list, select exactly {num_recommendations} articles that best match the inferred user interests.
3.  **Provide Explanations**: For each recommended article, write a concise (1-2 sentences) explanation. This explanation must clearly link the recommendation to the user's inferred interests or specific past readings.
4.  **Adhere to Output Format**: You *must* strictly follow the specified output format.
'''

    user_prompt = f'''User ID: {user_id}

### USER BROWSING HISTORY ###
{history_str}

### CANDIDATE NEWS ###
(Format: SequentialNumber. [Original_News_ID] "Title")
{candidate_str}

### YOUR TASK ###
Carefully analyze the USER BROWSING HISTORY to understand User {user_id}'s interests.
Then, from the CANDIDATE NEWS list, recommend exactly {num_recommendations} articles that are most relevant to these interests.
For each recommendation, provide a brief explanation.

### OUTPUT REQUIREMENTS (Follow Strictly) ###
Recommended news articles for User {user_id}:
1. [Original_News_ID_1] "Title of Recommended Article 1"
   Explanation: [Your explanation for article 1, linking to user interests or history]
2. [Original_News_ID_2] "Title of Recommended Article 2"
   Explanation: [Your explanation for article 2, linking to user interests or history]
... (continue for {num_recommendations} articles)

### CRITICAL REMINDERS ###
- **Source**: Recommendations *must* come *only* from the CANDIDATE NEWS list.
- **ID Format**: Use the original news ID (e.g., N12345) in brackets, not the sequential number.
- **Quantity**: Provide *exactly* {num_recommendations} recommendations. If fewer are genuinely suitable, recommend those and explicitly state why fewer are provided. If none are suitable, clearly state that.
- **Basis**: Your analysis and recommendations must be based *solely* on the information provided above.
'''
    
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    print(f"========== CHAT-RAC Recommendation for User {user_id} ({model}) ================")
    llm_output_text = personalized_generation(
        model=model,
        temperature=temperature,
        top_p=top_p,
        ctx=ctx,
        message=message,
        is_stream=is_stream
    )
    
    return {
        "llm_output": llm_output_text
    }


def evaluate_recommendations(results, ground_truth=None):
    """
    评估推荐系统的效率和准确性 (CHAT-RAC version)
    :param results: 推荐结果列表, 每个元素包含用户ID、推荐的LLM输出等.
    :param ground_truth: 真实标签，如果有的话.
    :return: 评估指标字典.
    """
    metrics = {
        "efficiency": {},
        "accuracy": {},
        "diversity": {},
        "coverage": {}
    }
    
    processing_times_list = [r["processing_times"]["total"] for r in results if "processing_times" in r and "total" in r["processing_times"]]
    if processing_times_list:
        metrics["efficiency"]["avg_total_time"] = np.mean(processing_times_list)
        metrics["efficiency"]["median_total_time"] = np.median(processing_times_list)
        metrics["efficiency"]["p95_total_time"] = np.percentile(processing_times_list, 95)
        if sum(processing_times_list) > 0:
            metrics["efficiency"]["throughput"] = len(results) / sum(processing_times_list)
        else:
            metrics["efficiency"]["throughput"] = float('inf') if len(results) > 0 else 0
    else:
        metrics["efficiency"]["avg_total_time"] = "N/A"
        metrics["efficiency"]["median_total_time"] = "N/A"
        metrics["efficiency"]["p95_total_time"] = "N/A"
        metrics["efficiency"]["throughput"] = "N/A"

    if ground_truth:
        all_candidate_news_set = set()
        all_recommended_news_set_overall = set() # For overall coverage

        for result_item_for_coverage in results: # Initial pass for overall coverage calculation
            if "candidate_news_ids" in result_item_for_coverage and result_item_for_coverage["candidate_news_ids"]:
                all_candidate_news_set.update(result_item_for_coverage["candidate_news_ids"])
            llm_output_text_cov = result_item_for_coverage.get("recommendations", {}).get("llm_output", "")
            current_candidate_ids_cov = result_item_for_coverage.get("candidate_news_ids", [])
            rec_news_ids_cov = extract_news_ids_from_recommendations(llm_output_text_cov, current_candidate_ids_cov)
            all_recommended_news_set_overall.update(rec_news_ids_cov)

        if all_candidate_news_set:
            metrics["coverage"]["item_coverage"] = len(all_recommended_news_set_overall) / len(all_candidate_news_set)
        else:
            metrics["coverage"]["item_coverage"] = 0.0

        for k_val in range(1, 11): # Calculate metrics for K from 1 to 10
            precision_at_k_list = []
            recall_at_k_list = []
            ndcg_at_k_list = []
            mrr_at_k_list = [] # MRR is typically @K but often reported as a single value based on first relevant item up to a K_max
            hit_rate_at_k_list = []
            
            for result_item in results:
                user_id = result_item["user_id"]
                
                if user_id in ground_truth and ground_truth[user_id]:
                    llm_output_text = result_item.get("recommendations", {}).get("llm_output", "")
                    current_candidate_ids = result_item.get("candidate_news_ids", [])
                    rec_news_ids = extract_news_ids_from_recommendations(llm_output_text, current_candidate_ids)
                    
                    actual_clicks = ground_truth[user_id]
                    
                    effective_recs = rec_news_ids[:k_val]
                    num_effective_recs = len(effective_recs)

                    if num_effective_recs > 0:
                        relevant_and_recommended = set(effective_recs).intersection(set(actual_clicks))
                        
                        precision = len(relevant_and_recommended) / num_effective_recs
                        recall = len(relevant_and_recommended) / len(actual_clicks) if actual_clicks else 0.0
                        
                        precision_at_k_list.append(precision)
                        recall_at_k_list.append(recall)
                        hit_rate_at_k_list.append(1 if len(relevant_and_recommended) > 0 else 0)
                        
                        # MRR: Calculated up to k_val
                        mrr_current_user = 0.0
                        for i, news_id_rec in enumerate(effective_recs):
                            if news_id_rec in actual_clicks:
                                mrr_current_user = 1.0 / (i + 1)
                                break
                        mrr_at_k_list.append(mrr_current_user)
                        
                        # NDCG: Calculated up to k_val
                        dcg = calculate_dcg(effective_recs, actual_clicks)
                        # For IDCG, consider only the top min(len(actual_clicks), num_effective_recs) items from actual_clicks
                        # as this is the best possible ranking up to num_effective_recs (which is <= k_val)
                        idcg = calculate_idcg(actual_clicks, num_effective_recs) 
                        ndcg = dcg / idcg if idcg > 0 else 0.0
                        ndcg_at_k_list.append(ndcg)
            
            if precision_at_k_list: # Check if any user had data for this K
                metrics["accuracy"][f"precision@{k_val}"] = np.mean(precision_at_k_list)
                metrics["accuracy"][f"recall@{k_val}"] = np.mean(recall_at_k_list)
                metrics["accuracy"][f"f1@{k_val}"] = calculate_f1(np.mean(precision_at_k_list), np.mean(recall_at_k_list))
                metrics["accuracy"][f"ndcg@{k_val}"] = np.mean(ndcg_at_k_list)
                metrics["accuracy"][f"hit_rate@{k_val}"] = np.mean(hit_rate_at_k_list)
                metrics["accuracy"][f"mrr@{k_val}"] = np.mean(mrr_at_k_list) # MRR@K for this specific K
            else: # Set to N/A if no data for this K
                for metric_name in ["precision", "recall", "f1", "ndcg", "hit_rate", "mrr"]:
                    metrics["accuracy"][f"{metric_name}@{k_val}"] = "N/A"

    diversity_scores = []
    for result_item in results:
        llm_output_for_diversity = result_item.get("recommendations", {}).get("llm_output", "")
        if llm_output_for_diversity:
            diversity = calculate_diversity(llm_output_for_diversity)
            diversity_scores.append(diversity)
    
    if diversity_scores:
        metrics["diversity"]["avg_diversity"] = np.mean(diversity_scores)
        metrics["diversity"]["intra_list_similarity"] = 1.0 - np.mean(diversity_scores) 
    else:
        metrics["diversity"]["avg_diversity"] = "N/A"
        metrics["diversity"]["intra_list_similarity"] = "N/A"
            
    return metrics


def extract_news_ids_from_recommendations(recommendations_text: str, candidate_news_ids: list[str] = None):
    """
    Extracts recommended news IDs from the LLM's text output, expecting a specific format.
    Format expected: "1. [NEWS_ID] \"Title\"\n   Explanation: ..."
    :param recommendations_text: The raw text output from the LLM.
    :param candidate_news_ids: Optional list of candidate news IDs. This is not strictly used for filtering 
                               if the LLM is expected to return original IDs, but can be useful for validation 
                               or if the LLM returns indices that need mapping (not the current CHAT-RAC design).
    :return: A list of extracted original news IDs.
    """
    if not recommendations_text or "can't fulfill" in recommendations_text.lower() or "unable to recommend" in recommendations_text.lower() or "cannot recommend" in recommendations_text.lower():
        print(f"解析推荐文本: LLM指示无法推荐或文本为空. 文本: {recommendations_text[:200]}...")
        return []

    recommended_ids = []
    # Regex to find lines like: "1. [N12345] ..." or "[N12345] ..." or "[N12345]"
    # It captures the news ID (alphanumeric, hyphens, underscores, periods).
    pattern = r"(?:\d+\.\s*)?\[([a-zA-Z0-9_.-]+)\]"
    
    # Debug: print(f"解析推荐文本: {recommendations_text}")
    
    try:
        matches = re.findall(pattern, recommendations_text)
        if matches:
            recommended_ids = matches # Takes all found IDs that match the pattern
            print(f"从推荐文本中解析出的新闻ID: {recommended_ids}")
        else:
            print(f"未能从推荐文本中通过主要模式解析出新闻ID. 文本内容:\n{recommendations_text[:500]}...")
            # Fallback: Attempt to find any IDs in square brackets if the primary pattern fails strictly
            # This might be too broad, consider if this fallback is truly needed or if prompt adherence is key
            # fallback_pattern = r'\[([a-zA-Z0-9_.-]+)\]' 
            # fallback_matches = re.findall(fallback_pattern, recommendations_text)
            # if fallback_matches:
            #     print(f"通过后备模式解析出的新闻ID: {fallback_matches}")
            #     recommended_ids = fallback_matches[:3] # Limit to 3 if using fallback

    except Exception as e:
        print(f"解析推荐文本时发生错误: {e}")
        return []

    # The prompt asks for a specific number of recommendations.
    # The calling function (e.g., in evaluation) will decide how many to use (e.g., K for P@K).
    # No explicit truncation to 3 here, as LLM should follow num_recommendations.

    return recommended_ids


def calculate_dcg(recommended, actual):
    """
    计算DCG (Discounted Cumulative Gain)
    :param recommended: 推荐的新闻ID列表
    :param actual: 用户实际点击/喜欢的新闻ID列表
    :return: DCG值
    """
    dcg = 0
    for i, item in enumerate(recommended):
        if item in actual:
            # 相关性为1，位置为i+1
            dcg += 1 / np.log2(i + 2)
    return dcg


def calculate_idcg(actual, k):
    """
    计算IDCG (Ideal Discounted Cumulative Gain)
    :param actual: 用户实际点击/喜欢的新闻ID列表
    :param k: 推荐列表长度
    :return: IDCG值
    """
    idcg = 0
    for i in range(min(len(actual), k)):
        idcg += 1 / np.log2(i + 2)
    return idcg


def calculate_f1(precision, recall):
    """
    计算F1分数
    :param precision: 精确率
    :param recall: 召回率
    :return: F1分数
    """
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def calculate_diversity(recommendations):
    """
    计算推荐结果的多样性
    :param recommendations: 推荐结果
    :return: 多样性分数
    """
    # 提取推荐文本中的关键词和主题
    import re
    from collections import Counter
    
    # 如果recommendations是字符串，直接使用；否则尝试转换为字符串
    if not isinstance(recommendations, str):
        try:
            recommendations = str(recommendations)
        except:
            return 0.5  # 默认值
    
    # 清理文本，只保留字母和空格
    text = re.sub(r'[^a-zA-Z\s]', ' ', recommendations.lower())
    
    # 移除常见的停用词
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 
                 'when', 'where', 'how', 'why', 'which', 'who', 'whom', 'this', 'that', 
                 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'to', 'from', 
                 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
                 'once', 'here', 'there', 'all', 'any', 'both', 'each', 'few', 'more', 
                 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 
                 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 
                 'now', 'news', 'article', 'articles', 'recommend', 'recommendation',
                 'recommendations', 'based', 'preferences', 'interest', 'interests',
                 'browsing', 'history', 'candidate', 'set', 'selected', 'similar'}
    
    # 分词并移除停用词
    words = [word for word in text.split() if word not in stop_words and len(word) > 3]
    
    # 计算词频
    word_counts = Counter(words)
    
    # 获取最常见的词（可能代表主题）
    common_words = word_counts.most_common(10)
    
    # 如果没有足够的词，返回默认值
    if len(common_words) < 2:
        return 0.5
    
    # 计算主题多样性：不同主题的数量除以总词数的比例
    unique_topics = len(common_words)
    total_topics = sum([count for _, count in common_words])
    
    # 计算基尼系数(Gini coefficient)来衡量多样性
    # 基尼系数越高，多样性越低；基尼系数越低，多样性越高
    topic_proportions = [count/total_topics for _, count in common_words]
    gini = sum([(i+1)*prop for i, prop in enumerate(sorted(topic_proportions, reverse=True))])
    gini = 2 * gini / len(topic_proportions) - (len(topic_proportions) + 1) / len(topic_proportions)
    
    # 将基尼系数转换为多样性分数（1-基尼系数）
    diversity_score = 1 - gini
    
    # 确保分数在0-1之间
    diversity_score = max(0, min(1, diversity_score))
    
    print(f"多样性评分: {diversity_score:.4f}, 主要主题: {[word for word, _ in common_words]}")
    
    return diversity_score


def multi_step_rec_batch(
        news_file: str = "/Users/john/CDU/recommend_sys/src/basic_skills/icl-rec/data/mind/news.tsv",
        behaviors_file: str = "/Users/john/CDU/recommend_sys/src/basic_skills/icl-rec/data/mind/behaviors.tsv",
        output_file: str = "/Users/john/CDU/recommend_sys/src/basic_skills/icl-rec/data/mind/recommendations.json",
        model: str = "gpt-4o",
        temperature: float = 0.2, 
        top_p: float = 0.9, 
        ctx: int = 13000,
        is_stream: bool = False,
        max_samples: int = 5, 
        evaluate: bool = True, 
        time_decay: bool = True, 
        max_retries: int = 2,
        num_recommendations_per_user: int = 3 # Added for CHAT-RAC
): 
    """
    批量处理用户数据并生成推荐 (CHAT-RAC version)
    :param news_file: 新闻数据文件路径
    :param behaviors_file: 用户行为数据文件路径
    :param output_file: 输出文件路径
    :param model: 使用的模型名称
    :param temperature: 温度参数
    :param top_p: top_p参数
    :param ctx: 上下文长度
    :param is_stream: 是否流式输出
    :param max_samples: 最大处理样本数量
    :param evaluate: 是否评估推荐结果
    :param time_decay: 是否考虑时间衰减因子 (影响历史新闻的顺序/选择)
    :param max_retries: 最大重试次数
    :param num_recommendations_per_user: 每个用户推荐的新闻数量
    :return: 推荐结果和评估指标
    """
    news_dict = {}
    print(f"正在读取新闻数据: {news_file}")
    with open(news_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            news_id = row[0]
            news_title = row[3]
            news_dict[news_id] = news_title
    print(f"共读取 {len(news_dict)} 条新闻数据")

    results = []
    sample_count = 0
    ground_truth = {}

    print(f"正在读取用户行为数据: {behaviors_file}")
    with open(behaviors_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if sample_count >= max_samples:
                break
                
            user_id = row[1]
            timestamp = row[2] if len(row) > 2 else None
            history = row[3]
            impression = row[4]
            
            history_list_ids = history.split(" ")
            impre_list_raw = impression.split(" ")
            
            # Filter for minimal data
            if len(history_list_ids) >= 1 and len(impre_list_raw) >= num_recommendations_per_user:
                print(f"\n处理用户 {user_id} 的数据 (样本 {sample_count + 1}/{max_samples})")
                
                history_news_titles = []
                history_news_with_weights = [] # For time decay logic
                
                for i, news_id_hist in enumerate(history_list_ids):
                    if news_id_hist in news_dict:
                        title = news_dict[news_id_hist]
                        history_news_titles.append(title)
                        if time_decay:
                            position = i + 1
                            weight = position / len(history_list_ids)
                            history_news_with_weights.append((title, weight))
                
                if time_decay and history_news_with_weights:
                    history_news_with_weights.sort(key=lambda x: x[1], reverse=True)
                    top_n = max(1, int(len(history_news_with_weights) * 0.7))
                    primary_interests = [item[0] for item in history_news_with_weights[:top_n]]
                    secondary_interests = [item[0] for item in history_news_with_weights[top_n:]]
                    history_news_titles = primary_interests + secondary_interests
                
                # history_news_titles is now ready

                candidate_news_with_ids = [] # List of (original_id, title)
                candidate_news_original_ids_for_eval = [] # For coverage calculation

                for impre_item in impre_list_raw:
                    impre_parts = impre_item.split("-")
                    if len(impre_parts) == 2:
                        impre_id, action = impre_parts
                        if impre_id in news_dict:
                            title = news_dict[impre_id]
                            candidate_news_with_ids.append((impre_id, title))
                            candidate_news_original_ids_for_eval.append(impre_id)
                            if int(action) == 1 and evaluate:
                                if user_id not in ground_truth:
                                    ground_truth[user_id] = []
                                ground_truth[user_id].append(impre_id)
                
                if history_news_titles and candidate_news_with_ids:
                    print(f"用户 {user_id} 有 {len(history_news_titles)} 条历史浏览记录和 {len(candidate_news_with_ids)} 条候选新闻")
                    
                    start_time = time.time()
                    recommendation_output_dict = None
                    llm_response_text = ""

                    for attempt in range(max_retries + 1):
                        try:
                            recommendation_output_dict = single_step_rec(
                                user_id=user_id,
                                history_news_titles=history_news_titles,
                                candidate_news_with_ids=candidate_news_with_ids,
                                model=model,
                                temperature=temperature,
                                top_p=top_p,
                                ctx=ctx,
                                is_stream=is_stream,
                                num_recommendations=num_recommendations_per_user
                            )
                            llm_response_text = recommendation_output_dict.get("llm_output", "")
                            rec_ids = extract_news_ids_from_recommendations(llm_response_text, candidate_news_original_ids_for_eval)
                            
                            if rec_ids and len(rec_ids) > 0:
                                break 
                            else:
                                print(f"尝试 {attempt+1}/{max_retries+1}: 推荐结果无效或未提取到ID，重试...")
                                if attempt == max_retries:
                                    print(f"所有重试均未能获得有效推荐。LLM原始输出: {llm_response_text[:200]}...")
                                    if not llm_response_text: # Ensure there is a default if LLM never responded
                                       recommendation_output_dict = {"llm_output": "无法生成有效的推荐 (LLM无响应)"}

                        except Exception as e:
                            print(f"尝试 {attempt+1}/{max_retries+1} 调用LLM失败: {str(e)}")
                            if attempt == max_retries:
                                print(f"所有尝试都失败: {str(e)}")
                                recommendation_output_dict = {"llm_output": f"无法生成有效的推荐: {str(e)}"}
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    result = {
                        "user_id": user_id,
                        "timestamp": timestamp,
                        "history_news_titles_provided": history_news_titles, # For reference
                        "candidate_news_details_provided": candidate_news_with_ids, # For reference
                        "candidate_news_ids": candidate_news_original_ids_for_eval, # For coverage metric
                        "recommendations": recommendation_output_dict, # This now holds {"llm_output": text}
                        "processing_times": {
                            "total": processing_time
                        }
                    }
                    results.append(result)
                    sample_count += 1
                    print(f"完成用户 {user_id} 的推荐，当前已处理 {sample_count}/{max_samples} 个样本")
                    print(f"处理时间: {processing_time:.2f}秒")
    
    print(f"\n将推荐结果保存到: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    metrics = {}
    if evaluate and results:
        print("\n评估推荐结果...")
        print(f"Ground Truth数据 (部分样本): {dict(list(ground_truth.items())[:3])}") # Print a few samples for check
        
        for res_item in results:
            user_id_eval = res_item["user_id"]
            llm_text_for_eval = res_item["recommendations"].get("llm_output", "")
            rec_news_ids_for_eval = extract_news_ids_from_recommendations(llm_text_for_eval, res_item["candidate_news_ids"])
            print(f"用户 {user_id_eval} 的推荐结果 (用于评估): {rec_news_ids_for_eval}")
            if user_id_eval in ground_truth:
                print(f"用户 {user_id_eval} 的真实点击: {ground_truth[user_id_eval]}")
            else:
                print(f"用户 {user_id_eval} 没有真实点击数据")
        
        metrics = evaluate_recommendations(results, ground_truth)
        metrics_file = output_file.replace('.json', '_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"评估指标已保存到: {metrics_file}")
        
        # 打印主要指标
        print("\n主要评估指标:")
        if "efficiency" in metrics and metrics["efficiency"]:
            for metric_name, display_name, suffix in [
                ('avg_total_time', '平均处理时间', '秒'), 
                ('throughput', '吞吐量', '用户/秒')
            ]:
                value = metrics['efficiency'].get(metric_name, 'N/A')
                if value == 'N/A' or isinstance(value, str):
                    print(f"{display_name}: {value}")
                else:
                    print(f"{display_name}: {value:.2f}{suffix}")
        
        if "accuracy" in metrics and metrics["accuracy"]:
            print("\n准确率指标 (P: Precision, R: Recall, F1: F1-Score, N: NDCG, H: Hit Rate, M: MRR):")
            header = "K  | P@K    | R@K    | F1@K   | N@K    | H@K    | M@K    "
            print(header)
            print("-" * len(header))
            for k_val in range(1, 11):
                p_k = metrics['accuracy'].get(f'precision@{k_val}', -1.0)
                r_k = metrics['accuracy'].get(f'recall@{k_val}', -1.0)
                f1_k = metrics['accuracy'].get(f'f1@{k_val}', -1.0)
                n_k = metrics['accuracy'].get(f'ndcg@{k_val}', -1.0)
                h_k = metrics['accuracy'].get(f'hit_rate@{k_val}', -1.0)
                m_k = metrics['accuracy'].get(f'mrr@{k_val}', -1.0)
                
                # Format to N/A if value is -1.0 (our placeholder for missing string 'N/A' from np.mean)
                # or if it's actually the string "N/A"
                p_k_str = f"{p_k:.4f}" if isinstance(p_k, float) and p_k != -1.0 else str(p_k if p_k != -1.0 else 'N/A')
                r_k_str = f"{r_k:.4f}" if isinstance(r_k, float) and r_k != -1.0 else str(r_k if r_k != -1.0 else 'N/A')
                f1_k_str = f"{f1_k:.4f}" if isinstance(f1_k, float) and f1_k != -1.0 else str(f1_k if f1_k != -1.0 else 'N/A')
                n_k_str = f"{n_k:.4f}" if isinstance(n_k, float) and n_k != -1.0 else str(n_k if n_k != -1.0 else 'N/A')
                h_k_str = f"{h_k:.4f}" if isinstance(h_k, float) and h_k != -1.0 else str(h_k if h_k != -1.0 else 'N/A')
                m_k_str = f"{m_k:.4f}" if isinstance(m_k, float) and m_k != -1.0 else str(m_k if m_k != -1.0 else 'N/A')

                print(f"{k_val:<2} | {p_k_str:<6} | {r_k_str:<6} | {f1_k_str:<6} | {n_k_str:<6} | {h_k_str:<6} | {m_k_str:<6}")
        
        if "diversity" in metrics and metrics["diversity"]:
            value_div = metrics['diversity'].get('avg_diversity', 'N/A')
            value_sim = metrics['diversity'].get('intra_list_similarity', 'N/A')
            print(f"多样性 (平均): {value_div if isinstance(value_div, str) else f'{value_div:.4f}'}")
            print(f"列表内相似度 (平均): {value_sim if isinstance(value_sim, str) else f'{value_sim:.4f}'}")
        
        if "coverage" in metrics and metrics["coverage"]:
            value = metrics['coverage'].get('item_coverage', 'N/A')
            print(f"覆盖率 (Item Coverage): {value if isinstance(value, str) else f'{value:.4f}'}")
    
    print(f"批量推荐完成，共处理 {sample_count} 个用户样本")
    return results, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='大模型新闻推荐系统 (CHAT-RAC adaptation)')
    
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'batch'],
                        help='运行模式: single(单个示例) 或 batch(批量处理)')
    parser.add_argument('--model', type=str, default='gpt-4o',
                        help='使用的大模型名称')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='温度参数')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='top_p参数')
    parser.add_argument('--ctx', type=int, default=13000,
                        help='上下文长度 (主要用于Ollama)')
    parser.add_argument('--stream', action='store_true', default=True, # Defaulting stream to True for single mode
                        help='是否流式输出')
    parser.add_argument('--max_samples', type=int, default=5,
                        help='批量处理模式下的最大样本数量')
    parser.add_argument('--news_file', type=str, 
                        default='/Users/john/CDU/recommend_sys/src/basic_skills/icl-rec/data/mind/news.tsv',
                        help='新闻数据文件路径')
    parser.add_argument('--behaviors_file', type=str,
                        default='/Users/john/CDU/recommend_sys/src/basic_skills/icl-rec/data/mind/behaviors.tsv',
                        help='用户行为数据文件路径')
    parser.add_argument('--output_file', type=str,
                        default='/Users/john/CDU/recommend_sys/src/basic_skills/icl-rec/data/mind/recommendations_single_step_after_tuning.json',
                        help='输出文件路径')
    parser.add_argument('--evaluate', action='store_true', default=True,
                        help='是否评估推荐结果 (仅batch模式)')
    parser.add_argument('--time_decay', action='store_true', default=True,
                        help='是否在批量模式中对历史记录考虑时间衰减')
    parser.add_argument('--max_retries', type=int, default=2,
                        help='LLM调用最大重试次数')
    parser.add_argument('--num_recommendations_per_user', type=int, default=3,
                        help='每个用户推荐的新闻数量')
    
    args = parser.parse_args()
    
    if args.mode == 'batch':
        print("启动 CHAT-RAC 批量处理模式...")
        multi_step_rec_batch(
            news_file=args.news_file,
            behaviors_file=args.behaviors_file,
            output_file=args.output_file,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            ctx=args.ctx,
            is_stream=args.stream, # Stream might be less useful for batch, but configurable
            max_samples=args.max_samples,
            evaluate=args.evaluate,
            time_decay=args.time_decay,
            max_retries=args.max_retries,
            num_recommendations_per_user=args.num_recommendations_per_user
        )
    else:
        # 单个示例模式 (CHAT-RAC)
        print("启动 CHAT-RAC 单个示例模式...")
        
        sample_user_id = "U123-SingleTest"
        sample_history_titles = [
            "'Wheel Of Fortune' Guest Delivers Hilarious, Off The Rails Introduction",
            "Hard Rock Hotel New Orleans collapse: Former site engineer weighs in",
            "Felicity Huffman begins prison sentence for college admissions scam",
            "Outer Banks storms unearth old shipwreck from 'Graveyard of the Atlantic'",
            "Tiffany's is selling a holiday advent calendar for $112,000"
        ]
        sample_candidate_news_with_ids = [
            ("N55555", "Browns apologize to Mason Rudolph, call Myles Garrett's actions 'unacceptable'"),
            ("N12345", "I've been writing about tiny homes for a year and finally spent 2 nights in a 300-foot home"),
            ("N23456", "Opinion: Colin Kaepernick is about to get what he deserves: a chance"),
            ("N34567", "The Kardashians Face Backlash Over 'Insensitive' Family Food Fight in KUWTK Clip"),
            ("N45678", "THEN AND NOW: What all your favorite '90s stars are doing today"),
            ("N88901", "Meghan Markle and Hillary Clinton Secretly Spent the Afternoon Together at Frogmore Cottage"),
            ("N67890", "Survivor Contestants Missy Byrd and Elizabeth Beisel Apologize For Their Actions"),
            ("N77777", "New tech gadgets review: Top picks for the holiday season"),
            ("N65432", "Exploring the future of sustainable architecture in urban environments"),
            ("N22334", "Celebrity chef shares easy recipes for Thanksgiving dinner")
        ]

        print(f"========== CHAT-RAC Single Example Call ({args.model}) ================")

        recommendation_result = single_step_rec(
            user_id=sample_user_id,
            history_news_titles=sample_history_titles,
            candidate_news_with_ids=sample_candidate_news_with_ids,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            ctx=args.ctx,
            is_stream=args.stream, # Stream is often True for single tests
            num_recommendations=args.num_recommendations_per_user
        )

        llm_output = recommendation_result.get("llm_output", "No output from LLM.")
        
        if not args.stream: # If not streaming, personalized_generation would have printed. If it is streaming, it prints char by char.
            print("\n--- LLM Recommendation Output (Single Mode) ---")
            print(llm_output)
            print("--- End of Output ---")
        
        # Optionally, parse and display IDs for single mode too
        extracted_ids = extract_news_ids_from_recommendations(llm_output, [item[0] for item in sample_candidate_news_with_ids])
        print(f"\nExtracted recommended IDs (single mode): {extracted_ids}")

        # Save single result to a file
        single_output_path = args.output_file.replace(".json", "_single_example.json")
        try:
            with open(single_output_path, 'w') as f:
                # For single mode, we can save the input and output for clarity
                single_result_to_save = {
                    "user_id": sample_user_id,
                    "history_provided": sample_history_titles,
                    "candidates_provided": sample_candidate_news_with_ids,
                    "llm_params": {
                        "model": args.model,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "num_recommendations": args.num_recommendations_per_user
                    },
                    "llm_output": llm_output,
                    "extracted_ids": extracted_ids
                }
                json.dump(single_result_to_save, f, indent=4, ensure_ascii=False)
            print(f"\nSingle example result saved to: {single_output_path}")
        except Exception as e:
            print(f"Error saving single example result: {e}")
