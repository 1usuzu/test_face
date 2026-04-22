# Release Readiness Checklist

Checklist này dùng trước khi push GitHub release candidate cho trạng thái `ai-v1.1-stabilized-c1`.

## Repository hygiene

- [ ] `git status` sạch hoặc chỉ chứa thay đổi dự kiến cho release.
- [ ] Không có file `.env` thật trong tracked files.
- [ ] Không có model weights local (`ai_deepfake/models/*.pth`) trong staged files.
- [ ] Không có absolute machine paths trong user-facing docs.
- [ ] Không có link local kiểu `file:///...` trong docs.

## AI module consistency

- [ ] README nêu đúng baseline governance:
  - 83.33% = historical only
  - 94.00% = reproducible technical anchor
- [ ] Threshold policy hiển thị rõ:
  - default `0.65`
  - high-recall `0.40`
  - high-precision `0.75`
- [ ] Source-of-truth operating policy tồn tại: `docs/c1_operating_points.json`.

## Backend readiness

- [ ] `backend/.env.example` liệt kê đầy đủ biến môi trường quan trọng.
- [ ] `backend/.env.production.example` phù hợp production.
- [ ] `/api/health` được document trong README.
- [ ] Startup command backend còn hợp lệ.

## Minimum validation before push

```bash
python -m pytest ai_deepfake/tests -v
python -m pytest backend/tests -v
python -c "import ast; ast.parse(open('ai_deepfake/detect.py', encoding='utf-8').read()); print('detect syntax OK')"
python -c "import ast; ast.parse(open('backend/api.py', encoding='utf-8').read()); print('api syntax OK')"
```

## Push gate

- [ ] Commit message(s) rõ scope release cleanup.
- [ ] Tag proposal đã thống nhất (ví dụ: `ai-v1.1-stabilized-c1`).
- [ ] Không còn known blocker mức high cho startup/deploy.
