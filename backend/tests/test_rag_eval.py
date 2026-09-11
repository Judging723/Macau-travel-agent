from eval.run_rag_eval import is_relevant_result, normalize_text


def test_normalize_text_unifies_traditional_and_simplified_chinese() -> None:
    assert normalize_text("澳門博物館") == normalize_text("澳门博物馆")


def test_relevant_result_accepts_an_alternative_source() -> None:
    case = {
        "expected_source": "澳门世界遗产",
        "relevant_passages": [
            {
                "source": "澳门满Fun社区步行路线",
                "page_number": 1,
                "keywords": ["东望洋灯塔", "1864年"],
            }
        ],
    }
    result = {
        "title": "澳门满Fun社区步行路线",
        "page_number": 1,
        "content": "东望洋灯塔建于1864年。",
    }

    assert is_relevant_result(result, case)
