"""主动引导式推荐冒烟测试

场景：某青少年爱看暴力血腥动作片 → 引导到同等刺激度的非血腥电影。
输出：比较 α=1.0（纯用户兴趣）、α=0.5（平衡）、α=0.2（偏主题）三种设置下的推荐差异。
"""
import sys
import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.guided_recommender import build_default_recommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    rec = build_default_recommender(PROJECT_ROOT, with_llm=True)

    user_movies = [
        "Die Hard", "Terminator 2: Judgment Day", "Rambo III",
        "Predator", "Pulp Fiction", "Reservoir Dogs",
    ]
    target = "不含血腥的惊险刺激"
    avoid = ["血腥", "露骨暴力"]

    print("\n======== 输入 ========")
    print(f"用户兴趣（电影）：{user_movies}")
    print(f"目标主题：{target}")
    print(f"避免元素：{avoid}")

    results = {}
    for alpha in [1.0, 0.5, 0.2]:
        print(f"\n======== α = {alpha} ========")
        r = rec.recommend(
            target_keyword=target,
            movie_titles=user_movies,
            alpha=alpha,
            top_k=5,
            faiss_top_k=50,
            avoid_terms=avoid,
            use_llm_rerank=True,
        )
        print(f"[画像] {r.user_profile[:120]}...")
        print(f"[主题] {r.topic_description[:120]}...")
        print(f"[推荐 Top-5]")
        for i, x in enumerate(r.recommendations, 1):
            print(f"  {i}. {x['title']} ({x['year']}) - {x['genres']}")
            print(f"     对用户={x['score_user']:.3f}  对主题={x['score_topic']:.3f}  融合={x['score_fused']:.3f}")
            if x.get("reason"):
                print(f"     理由: {x['reason']}")
        results[str(alpha)] = r.to_dict()

    out = PROJECT_ROOT / "results" / "cases" / "guided_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
