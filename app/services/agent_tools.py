"""
Agent 工具定义。

每个工具包含名称、描述和 JSON Schema 参数定义，
供 LLM Agent 在决策循环中调用。
"""

TOOL_DEFINITIONS = [
    {
        "name": "search_movies",
        "description": (
            "搜索电影数据库，返回符合筛选条件的候选电影列表。"
            "支持按类型、年份、评分、评分人数等维度筛选。"
            "每次调用返回 top_k 部候选电影及其元数据。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "语义搜索文本，用于向量检索和 BM25 关键词匹配。应基于用户需求构建英文搜索短语。",
                },
                "genre": {
                    "type": "string",
                    "description": (
                        "包含的电影类型，多个用逗号分隔，如 'Science Fiction,Thriller'。"
                        "设置为 null 则不限制类型。"
                    ),
                },
                "exclude_genre": {
                    "type": "string",
                    "description": "排除的电影类型，如 'Horror'。设置为 null 则不排除。",
                },
                "year_from": {
                    "type": "integer",
                    "description": "上映年份下限（含），如 2010 表示 2010 年及以后。设置为 null 则不限制。",
                },
                "year_to": {
                    "type": "integer",
                    "description": "上映年份上限（含），如 2000 表示 2000 年及以前。设置为 null 则不限制。",
                },
                "min_vote_average": {
                    "type": "number",
                    "description": "最低 TMDB 评分（0-10），如 7.0。设置为 null 则不限制。",
                },
                "min_vote_count": {
                    "type": "integer",
                    "description": "最低 TMDB 评分人数，如 500。设置为 null 则不限制。",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的候选数量，默认 200。",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_user_profile",
        "description": "获取指定用户的观影偏好画像，包含偏好类型、平均评分等。",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "integer",
                    "description": "用户 ID",
                },
            },
            "required": ["user_id"],
        },
    },
]


def execute_tool(
    tool_name: str,
    tool_params: dict,
    retrieval_service=None,
    user_profile_service=None,
    user_seen_movies_service=None,
) -> dict:
    """
    执行工具调用。

    Args:
        tool_name: 工具名称
        tool_params: 工具参数
        retrieval_service: RetrievalService 实例
        user_profile_service: UserProfileService 实例
        user_seen_movies_service: UserSeenMoviesService 实例

    Returns:
        工具执行结果
    """
    if tool_name == "search_movies":
        if retrieval_service is None:
            return {"error": "retrieval_service not available"}

        return retrieval_service.search_movies(
            query=tool_params.get("query", ""),
            top_k=tool_params.get("top_k", 200),
            genre=tool_params.get("genre"),
            exclude_genre=tool_params.get("exclude_genre"),
            year_from=tool_params.get("year_from"),
            year_to=tool_params.get("year_to"),
            min_vote_average=tool_params.get("min_vote_average"),
            min_vote_count=tool_params.get("min_vote_count"),
        )

    if tool_name == "get_user_profile":
        if user_profile_service is None:
            return {"error": "user_profile_service not available"}
        user_id = tool_params.get("user_id")
        profile = user_profile_service.get_profile(user_id)
        if profile is None:
            return {"error": f"User {user_id} not found"}
        return {"profile": profile}

    return {"error": f"Unknown tool: {tool_name}"}
