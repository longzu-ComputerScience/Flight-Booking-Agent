# Flight Booking Agent — BTVN#3 Agentic AI

Agent đặt vé máy bay **mô phỏng** bằng LangChain/LangGraph, triển khai bốn lớp harness (ràng buộc dữ liệu, kiểm hoàn thành bằng code, kiểm quyền, bàn giao), safety guard và ba mẫu thiết kế ReAct, Plan-then-Execute, Hybrid.

## Kiến trúc

`User → Model → Architecture graph → Runtime/Harness → Tool → Observation/State → tiếp tục hoặc Completion/Handoff`

Model đề xuất hành động; LangGraph điều phối; runtime kiểm schema, quyền và ràng buộc trước khi thực thi tool. State giữ kết quả trong bộ nhớ theo từng lần chạy. Chỉ được báo DONE khi booking và bản tóm tắt vượt qua kiểm tra provenance bằng code. Budget, phát hiện lặp và thiếu tiến triển giúp giới hạn quá trình chạy.

Bốn tool: `search_flights`, `get_flight_details`, `hold_flight`, `book_flight`. Dữ liệu và inventory đều là mock, không mua vé hoặc thanh toán thật. Quyền booking đến từ xác nhận ngoài model, có phạm vi tuyến, ngày, hành khách và ngân sách.

| Kiến trúc | Cách điều phối | Đặc điểm |
|---|---|---|
| ReAct | Decide → act → observe → decide | Phản ứng theo observation, nhiều lần gọi model |
| Plan-then-Execute | Lập kế hoạch → bind tham chiếu → execute; replan khi lỗi | Ít gọi model với quy trình ổn định |
| Hybrid | Kế hoạch discover/reserve/finalize + quyết định cục bộ | Kiểm soát theo stage, thêm overhead |

## Cấu trúc

```text
Flight-Booking-Agent/
├── .gitignore
├── .env.example
├── README.md
├── requirements.txt
├── main.py
└── app/
    ├── __init__.py
    ├── config.py
    ├── state.py
    ├── models.py
    ├── runner.py
    ├── data/
    ├── tools/
    ├── harness/
    ├── agents/
    └── evaluation/
```

## Cài đặt và chạy

Python 3.11 trở lên. Trong PowerShell, tại thư mục project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python main.py --architecture react --model fake
python main.py --architecture plan --model fake
python main.py --architecture hybrid --model fake
python main.py --evaluate --model fake --today 2026-10-06
```

Demo mặc định dùng fixture có xác nhận booking. Ngày bay mặc định là ngày mai theo giờ Việt Nam; dùng `--today YYYY-MM-DD` để cố định ngày tham chiếu. Yêu cầu riêng qua `--request` chưa có quyền booking nếu không cung cấp xác nhận có scope; xem `python main.py --help`.

Evaluation chạy 13 scenario × 3 kiến trúc, gồm cả lỗi và tình huống an toàn; tạo `results/evaluation.json` và `results/evaluation.md` trên máy. `success` là booking hoàn thành; `scenario_pass` là kết quả đúng kỳ vọng, có thể là HANDOFF. Fake model deterministic giúp so sánh orchestration, không đo chất lượng LLM thật. State chỉ lưu trong RAM; dùng `--json-output results/run.json` nếu cần xuất kết quả một lần chạy.

## Model thật (tùy chọn)

Sao chép `.env.example` thành `.env`, đặt `SE373_MODEL=provider:model` với model hỗ trợ tool calling và API key tương ứng. Cài integration theo provider: `langchain-openai`, `langchain-anthropic` hoặc `langchain-google-genai`; provider lần lượt là `openai`, `anthropic`, `google_genai`.

```powershell
Copy-Item .env.example .env
# Cấu hình .env và cài integration tương ứng trước khi chạy.
python main.py --architecture react --model real
```

Không hard-code hoặc commit API key. Model thật vẫn dùng mock tools và cùng harness; các kết quả evaluation mặc định chỉ được đo với fake model.

---

## Author profile and links

For more projects and background, see [my GitHub profile](https://github.com/longzu-ComputerScience) and [LinkedIn](https://www.linkedin.com/in/vuz-long-7b4744393/). This profile link identifies my account and does not imply sole authorship of a group project.
