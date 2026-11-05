# license-donor-mask

運転免許証の**裏面**写真から「臓器提供に関する意思表示」欄だけを黒く塗りつぶす、ローカル完結のマスキングツール（Python / OpenCV）。
設計の主眼は「塗れること」ではなく **「確信が持てないときは塗らず、人手レビューに回す（fail-closed）」** と、
**「正しい場所を塗ったことを、塗った後に独立して検証する」** こと。

> **状態: v1（参考実装）**。幾何ベースの検出・fail-closed・マスク後検証・テストは動作しますが、
> 同梱のテンプレートは **合成（SYNTHETIC_TEST）レイアウト用だけ**です。**実物の免許証様式のテンプレートは未作成**（校正が必要）なので、
> 実画像を入れると現状は **すべて review に回ります**（後述「実画像での検証結果」）。「そのまま本番投入できる」ものではありません。

- ライセンス: MIT（`LICENSE`）
- 設計の詳細: [`DESIGN.md`](DESIGN.md)（パイプライン、しきい値、校正手順、fail-closed 条件、検証方法、設計書との差分）

## 何をするか

1. カードの四辺を検出して台形補正（正規化キャンバス 1011×638px = 85.6×54mm）。ここで **面積最大では選ばない**。
2. 4回転 × 鏡像 = 8通りの向き仮説を、様式テンプレートとの罫線レイアウト照合（ZNCC）で評価し、**一意に決まったときだけ**採用。
3. 欄を3つの独立した手がかり（T: テンプレート位置 / F: 罫線の枠 / A: 選択肢の印の3連）で特定し、**3つが一致したときだけ**マスク範囲を決める。
4. 単色（黒、または白）で塗り、**PNG** で書き出す前に、6項目の検証（V1〜V6）を全部通す。1つでも落ちたら `ok/` には何も出さない。

### なぜ AI・外部送信を使わないか

臓器提供の意思は要配慮個人情報です。画像を外部 API や LLM に送れば、それ自体が漏えい経路になります。
また、機械学習モデルの出力は説明しにくく、同じ入力でも環境差で変わりえます。本ツールは

- ML モデル・学習済み重み（`cv2.dnn` を含む）・OCR・LLM・クラウド API を**一切使わず**、OpenCV の決定的処理だけで完結します。
- ネットワークアクセスのコードを持ちません（テストで `socket` / `urllib` / `requests` 等の import が無いことを確認）。
- 乱数を使いません。`cv2.setNumThreads(1)`、PNG 圧縮パラメータ固定、入力はソート順。同じ入力・同じ設定なら出力 PNG の sha256 とログ（時刻・処理時間を除く）が一致します（テストで確認）。
- 判定は「どのゲートがどの値で落ちたか」がログ JSON にそのまま残ります（スコアの合成はしない）。

## fail-closed 方針

成功（`ok`）は **全ゲート通過の AND のみ**。次はどれも「処理成功」にせず、`out/review/` に原本へ検出結果を重ねた画像を出し、
理由コード付きのログを残して終了コード 3 にします（マスク済み画像は出力しません）。

| 状況 | 理由コード（例） |
|---|---|
| カード四辺が4点として取れない（**「画像全体をカードとみなす」フォールバックは存在しません**） | `C_NO_CARD` |
| カードが画像いっぱい（余白ゼロで撮影。画像全体の輪郭だけがカード比 1.585±5% の四角形）→ **再撮影を案内**（条件付きフォールバックは入れていない） | `C_FRAME_FILLED` |
| 遠近が強すぎる（対辺比 > 1.30）／端で欠けている／マップ間の合意なし | `C_PERSPECTIVE` / `C_TOUCHES_BORDER` / `C_NO_CONSENSUS` |
| 候補が複数あり一意に決まらない（台紙・2枚写り。**面積最大を自動採用しない**） | `O_AMBIGUOUS` / `C_MULTIPLE_CARDS` |
| 表面・他カード・鏡像 | `O_FRONT_SIDE` / `O_UNSUPPORTED_CARD` / `O_MIRRORED` |
| 向き（8仮説）が一意に決まらない・同点（差 1e-6 以内）・照合スコア不足 | `O_AMBIGUOUS` / `O_LOW_SCORE` |
| 精密位置合わせ（ECC）が外れる | `R_ECC_FAIL` |
| 反射・ぼけ・低コントラスト・低解像度 | `Q_GLARE` / `Q_BLUR` / `Q_CONTRAST` / `Q_LOW_RES` |
| 欄の手がかり（枠・チェックボックス3連）が無い／複数／互いに不一致 | `F_FRAME_NOT_FOUND` / `F_ANCHOR_NOT_FOUND` / `F_ANCHOR_MULTIPLE` / `F_CUE_DISAGREE` |
| 保護領域との衝突・マスク面積異常・手書きのはみ出し | `M_PROTECTED_CONFLICT` / `M_AREA_OUT_OF_RANGE` / `M_INK_OVERFLOW` |
| 塗った後の検証（V1〜V6）のどれかが不合格 | `V_INCLUSION` / `V_FILL_NOT_UNIFORM` / `V_OUTSIDE_CHANGED` / `V_CARD_DAMAGED` / `V_REDETECT_TRACE` / `V_BOUNDARY_INK` |
| 各段で例外・shadow モード | `X_EXCEPTION`（握り潰して ok にしない）/ `X_SHADOW` |
| 入力 | `E_FORMAT_UNSUPPORTED`（JPEG/PNG 以外）/ `E_TOO_LARGE`（50MP 超）/ `E_DECODE` |

### 塗った後の検証（全部 AND。`donor_mask/verify.py`）

| | 内容 |
|---|---|
| V1 位置正当性 | 検出した F・A・T がマスク内側（正規化座標で 6px 以上）にあり、マスクが `field_rect` を100%含み IoU ≥ 0.70。**「塗れたか」ではなく「正しい場所か」を見る**（試作は一様性しか見ておらず、位置がずれていても verified=true になった） |
| V2 一様性 | マスク内が全て塗り色と完全一致（メモリと、書き出した PNG の読み戻しの両方） |
| V3 欄外不変 | マスク外が原本のデコード配列と完全一致（差分の非ゼロ画素数 0、読み戻し後） |
| V4 再検出 | 出力に再度カード検出とチェックボックス検出を当て、カード四角が壊れておらず、窓内に印が残らず、マスク内（3px 侵食）に Canny エッジが 0 |
| V5 境界インク | マスク外周リングに手書きのはみ出しが残っていない |
| V6 原本不変 | 処理後に原本の sha256 と mtime が処理前と一致（不一致ならバッチ全体を停止） |

出力は `ok/.staging/` に masked と preview を書き、masked は読み戻して V2/V3、preview はデコード可能なことを確認してから V6・原本不変を通し、2つ揃ってから `os.replace` で `ok/` へ確定します。確定後に失敗したら ok/ の分を削除してから review、削除も失敗したら error でバッチ停止します。

## 使い方

Python 3.11 以上。

```bash
python3.11 -m venv .venv && . .venv/bin/activate     # または: uv venv --python 3.11
pip install -r requirements-dev.txt                    # 本番依存は opencv-python-headless と numpy のみ

# マスク（ファイル1つ、またはディレクトリ直下の jpg/jpeg/png を一括。ソート順）
python -m donor_mask run --in ./inbox --out ./out                 # 既定: 黒
python -m donor_mask run --in ./inbox --out ./out --fill white
python -m donor_mask run --in ./inbox --out ./out --shadow        # ok 相当でも ok/ に出さず全件 review（提案を画像に描く）

# 校正用: 正規化画像と 5% グリッド焼き込み画像（出力先はアクセス制限ディレクトリに）
python -m donor_mask rectify --in ./sample.jpg --out ./calib

# テンプレート作成（設計書 5.4 C1）: 正規化済みの空欄サンプル＋測定値JSON から templates/<id> を作る
#   測定値JSON: field_rect / anchor_type / anchor / required_cues / protected_rects / ignore_rects
#               （座標は推測しない。5枚未満は既定でエラー、--allow-fewer で暫定扱い＝min_samples_met=false）
python -m donor_mask template-build --in ./calib_samples --out ./templates/my_back_v1 --id my_back_v1 \
    --label back --spec ./my_back_v1_spec.json

# 独立再検証（原本・マスク済みPNG・ログから V1〜V6 をやり直す）
python -m donor_mask verify --orig ./inbox/a.jpg --masked ./out/ok/<job_id>__masked.png --log ./out/logs/<job_id>.json
```

`--config`（既定 `config.toml`）と `--templates`（既定 `templates/`）で差し替えられます。しきい値は全て `config.toml`、設定ファイルの sha256 は毎回ログに残ります。

出力の構成:

```
out/ok/{job_id}__masked.png      マスク済み画像（全ゲート通過のものだけ。PNG のみ）
out/ok/{job_id}__preview.jpg     マスク後の画像にマスク枠（赤）とカード四辺（緑）。欄の中身は写らない
out/review/{job_id}__review.jpg  原本に検出結果と理由コードを重ねたもの（欄の中身が見える＝取扱注意）
out/logs/{job_id}.json           1件ごとのログ（数値と幾何のみ。画素・欄の内容は入れない）
out/logs/batch_*.json            バッチ集計
```

- `job_id` は入力バイト列の sha256 先頭16桁。**元ファイル名は出力名に使いません**（氏名を含みうるため）。
- 出力先が入力ディレクトリと同じ／入力の配下なら起動エラー（終了コード 1）。既存の出力は上書きせず、その件を `X_OUTPUT_EXISTS` にします。原本は読み取り専用で開きます。
- EXIF は再エンコードで落ちます（意図した挙動。EXIF の向きは読み込み時に反映）。
- 終了コード: `0` 全件 ok / `3` review（または件単位の error）を含む / `1` 設定・テンプレート不正や I/O の致命的エラー、原本が処理中に変化した場合。
- `--fill noise`、JPEG 出力、ぼかし・モザイク・半透明は**実装していません**（秘匿性は上がらず、「欄外の差分ゼロ」「塗りの一様性」を厳密に検証できなくなるため）。

## テスト

```bash
pytest -q                      # 合成データのみ。オフライン。実在の免許証画像は使いません
DONOR_MASK_SAMPLE_DIR=/path/to/samples pytest tests/test_real_samples.py -s   # 任意: 実見本での挙動確認（見本は同梱しません）
```

画像は全て `tests/fixtures_gen.py` が**決定的に生成する合成データ**（`SYNTHETIC_TEST` と明記）で、コミットしません。固定された変換グリッドの一部（回転 0/±5/±15/90/180/270、縮小、余白、遠近、γ、ぼかし、JPEG 品質 70/95、背景 3 種、台紙の4アスペクト、鏡像 ほか）を扱います。

テストが固定していること（抜粋）:

- 陽性: ok になった全件で **包含率 100%（GT_field を1px侵食しても全画素が塗られる）・欄外差分 0・保護領域 100% 一致・塗りすぎ ≤ 1.6 倍**。1件でも破れたらテスト失敗（＝誤通過）。
- 陰性: 台紙（アスペクト 1.19/1.40/1.585/1.98）で紙を選ばない／上辺が下辺の 50/60/70% の斜めは review／カードなし・端で欠け・2枚・表面・別様式・鏡像・指（肌色の楕円）・反射・σ=4 のぼけ・枠だけでチェックボックス無し・3連が2組 は review で、`ok/` が空。
- 各段（S1〜S7、検証）に例外を注入 → `X_EXCEPTION` で review。S6 の結果を正規化座標で 30px ずらす注入 → V1（`V_INCLUSION`）で review（V2 は通るので、一様性だけの検証だと通っていたケース）。
- `verify.py` 単体: 一様でない／欄外が1画素違う／位置がずれた画像をそれぞれ落とす。`verify.py` は `mask.py` / `pipeline.py` / `orient.py` を import しない。
- 再現性・原本不変（sha256 と mtime）・既存出力の非上書き・入出力ディレクトリ検査・EXIF・アルファ付き PNG。

## 実画像での検証結果

以下の2枚で試しました。**どちらも実物の免許証の写真ではなく、「記入例」「解説図」です**（個人情報は含まれません）。権利の都合でリポジトリには含めません。出典:

- 熊本県移植医療推進財団「意思表示について」ページ掲載の「運転免許証（裏面）の記入例」: <https://kumaisui.jp/intent/>（`kumaisui_driver_license.png`, 340×213）
- 厚生労働省「運転免許証の様式変更について」ページ掲載の解説図: <https://www.mhlw.go.jp/bunya/kenkou/zouki_ishoku/intention_menkyosyou.html>（`mhlw_intention_01.jpg`, 520×480。左に改正前・右に改正後の裏面と、臓器提供欄の拡大図を並べた図）

結果（v1・既定設定・`python -m donor_mask run`、原本の sha256/mtime は不変）:

| 見本 | カード検出 | 向き | 検出された欄 | 被覆率 | verified | 結果 |
|---|---|---|---|---|---|---|
| kumaisui_driver_license.png | 未実施（S0 で停止） | — | — | — | — | **review**（終了コード 3）`Q_LOW_RES`: 長辺 340px < 800px |
| mhlw_intention_01.jpg | 未実施（S0 で停止） | — | — | — | — | **review** `Q_LOW_RES`: 長辺 520px < 800px |

解像度ゲートを診断用に 200px へ下げた実行（`min_long_edge_px = 200` の一時設定。既定値は変えていません）:

| 見本 | カード検出 | 向き | 検出された欄 | 被覆率 | verified | 結果 |
|---|---|---|---|---|---|---|
| kumaisui_driver_license.png | 候補 0 件（画像いっぱいの輪郭は枠いっぱい判定 `C_FRAME_FILLED` または `C_AREA`、内部の区画は `C_ASPECT` ×5 で棄却） | — | — | — | — | **review** `C_NO_CARD` |
| mhlw_intention_01.jpg | 候補 0 件（`C_AREA` ×1、`C_ASPECT` ×2） | — | — | — | — | **review** `C_NO_CARD` |

（`C_FRAME_FILLED` は画像のアスペクトがカード比 1.585±5% のときだけ付く。画像全体の輪郭はどの写真にも現れるので、他に候補が取れなかったときに限り理由コードにする。）

どちらも**マスク画像は出力されません（`ok/` は空）。これは設計どおりの正しい挙動**です。見本は撮影された写真ではなく（kumaisui は背景が無く、画像の縁がカード外形にほぼ一致する切り出し〔角は透過〕、mhlw は複数の図を並べた解説図）、
カード四辺を背景との境界から一意に取れる入力ではありません。無理に通すフォールバックは作っていません。
つまり、**これらの実画像でマスクが成功することは、この v1 では確認できていません**（検出が正しく動くか・欄の座標が合うかは、実物の様式テンプレートを作って校正しないと分かりません）。

## 既知の限界（v1）

- **実物様式のテンプレートが無い。** 同梱は合成レイアウト（`templates/synthetic_*`）のみ。テストが示すのは「想定レイアウトなら、回転・遠近・台紙・上下逆などでも静かに誤マスクしない」ことまでで、実物の様式が想定どおり（欄に枠罫線がある／選択肢の印が四角3個の縦並び など）かは**未確認**。チェックボックス前提は仮説で、○で囲む方式向けの検出器（`anchor_type = "circle3"`）は用意したが実物では未検証。
- 欄の座標・保護領域・無視領域・各しきい値は全て「初期値」。実物での校正（DESIGN.md 5 章）が必要。
- **未実装のサブコマンド**: `calibrate`（陽性・陰性分布からのしきい値提案）、`make-fixtures`（生成器は `tests/fixtures_gen.py` にあり、CLI 化していない）。呼ぶと終了コード 1 と「未実装」を返す。`labels.jsonl`・コンタクトシートによる ok 全件目視も未実装。`template-build`（設計書 5.4 C1）は実装済み（正規化済みサンプルの ECC 平均から `ruling.png` / `print.png` を作り、測定値は `--spec` で必ず与える。5枚未満は既定でエラー、`--allow-fewer` で `min_samples_met=false` の暫定テンプレート）。
- V4 の再検出は、カード検出（合意マップ 1・縁の支持率 0.5 に緩和）とチェックボックス検出・窓内エッジの確認まで。S1〜S6 の完全な再実行ではない（DESIGN.md 参照）。
- 指かぶり・部分遮蔽は専用検出器を持たず、枠の支持率・アンカー有無・コントラストで落とす。モアレ・コピー画像は個別には検出しない。
- 反射・ぼけ・コントラスト・解像度のしきい値、カード面積比 0.15 以上などは撮影条件に依存し、実データで要校正（例: スマホ写真でカードが画像の15%未満だと `C_AREA` で review）。
- マイナンバーカード（マイナ免許証）は対象外。投稿された場合は個人番号が写るので、受領ポリシーの確認が別途必要。
- レビュー担当者は原本（要配慮個人情報）を見る。`out/review/` のアクセス制限と保存期間は運用側で決めること。

## ライセンスと権利

MIT License, Copyright (c) 2026 Masayoshi Michikawa。見本画像（上記2枚）の権利は各権利者に帰属し、このリポジトリには含まれません。
