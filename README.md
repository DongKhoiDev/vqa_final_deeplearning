# Traffic Sign Visual Question Answering

Đồ án cuối kỳ môn Học sâu: hệ thống hỏi đáp ảnh biển báo giao thông tiếng Việt.

## Nội dung chính

- `source.ipynb`: notebook chính để kiểm tra dataset, train/evaluate A1/A2, chạy B1/B2 Qwen2-VL và demo Gradio trên Colab.
- `server.py`: FastAPI backend cho demo local.
- `demo.html`: giao diện demo local.
- `dataset/`: dữ liệu VQA biển báo giao thông.
- `results/`: prediction và metric của B1/B2.
- `checkpoints/b2_qwen_lora_safe/`: LoRA adapter B2.

## Dataset

| Split | Số ảnh | Số QA |
|---|---:|---:|
| Train | 206 | 2060 |
| Validation | 25 | 250 |
| Test | 27 | 270 |

Tổng dữ liệu gồm 258 ảnh và 2580 cặp hỏi đáp.

## Mô hình

- **A1**: kiến trúc rời với LSTM decoder.
- **A2**: kiến trúc rời với Transformer decoder.
- **B1**: Qwen2-VL zero-shot.
- **B2**: Qwen2-VL fine-tuned bằng LoRA.

## Kết quả chính

| Model | Metric |
|---|---:|
| A1 | Val Acc: 66.78% |
| A2 | Val Acc: 74.97% |
| B1 | Exact Match: 2.96% |
| B2 | Exact Match: 43.33% |

## Chạy demo local

```powershell
pip install fastapi uvicorn torch torchvision pillow pydantic
python server.py
```

Sau đó mở:

```text
http://127.0.0.1:8000/
```

Demo local hỗ trợ A1/A2 realtime. B1/B2 local dùng prediction đã lưu trong `results/`.

## Demo B1/B2 realtime trên Colab

Trong `source.ipynb`:

```python
USE_COLAB_DRIVE = True
RUN_B2_LORA_TRAIN = False
RUN_B2_EVAL = False
```

Sau đó chạy các cell load B2 và cell Gradio ở cuối notebook.

## Checkpoint lớn

Các file checkpoint `.pt` lớn không được đưa lên GitHub do giới hạn dung lượng. Chúng nên được chia sẻ bằng Google Drive hoặc Hugging Face Hub.

