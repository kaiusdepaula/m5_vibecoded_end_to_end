Place the M5 extract here before running `docker compose up --build`:

- `sales_train_validation.csv` (or equivalent `sales` extract)
- `calendar.csv`
- `sell_prices.csv`

This directory is bind-mounted read-only into the `train` service at `/app/data`.
