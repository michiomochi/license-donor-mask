# license-donor-mask

運転免許証の**裏面**の写真から、臓器提供の意思表示欄を**黒く塗る**ツールです。

- 外部送信なし。AI・OCR・学習モデルは使わず、OpenCV の決まった処理だけで動きます。
- 自信がないときは塗らず、人の確認待ち（review）に回します。
- **状態**: 手元のテスト画像 9 枚（免許証 7 枚）でしか確認していません。実写真での精度は未確認です。

## 使い方

```bash
# 初回のみ（Python 3.11 以上）
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 実行（--in は写真のフォルダ。元の写真は変更しません）
python -m donor_mask run --in ./inbox --out ./out --config config.tolerant.toml
```

## 結果

- `out/ok/*__masked.png`: 塗れた画像。**次の工程に渡すのはこれだけ**です。
- `out/review/`: 塗れなかった画像と理由。人が確認するか、撮り直しを依頼します（欄の中身が見えるので取扱注意）。
- `out/logs/`: 1 件ごとの数値の記録。
- 終了コード: `0` 全件 ok / `3` review あり / `1` 致命的エラー。

## 設定

| `--config` | 内容 |
| --- | --- |
| `config.tolerant.toml`（推奨） | 黒塗りの位置のずれを許容。個人情報の塗り残しは許容しない |
| 指定なし（`config.toml`） | 厳密。迷ったら review。review が多くなる |

## 本番で確認するとき

```bash
python -m donor_mask aggregate --logs ./out --mode run
```

ok / review の件数と、review の理由別の件数だけが出ます（画像・ファイル名は出ません）。共有はこの集計だけにしてください。

## 詳細

- [docs/REFERENCE.md](docs/REFERENCE.md): 理由コード一覧、判定の仕組み、検証結果、既知の限界、他のコマンド
- [DESIGN.md](DESIGN.md): 設計
- MIT License（[LICENSE](LICENSE)）
