# DESIGN — 運転免許証裏面「臓器提供に関する意思表示」欄マスキング（v1 実装）

このファイルは **実装 v1 の設計**です。元になった設計書（`donor_field_masking_design.md`。(A) 設計書と (B) 実装依頼プロンプト）の章番号に合わせてあり、
**設計書との差分・未実装は 8 章にまとめています**。数値・座標は断定せず、「初期値」と書いたものは校正で決める値です（`config.toml` で変更、sha256 がログに残る）。

## 0. 原則

| | |
|---|---|
| P1 | 面積最大で選ばない。幾何条件を通った候補は全部残し、罫線レイアウトのテンプレート照合（ZNCC）とマージンで選ぶ。2つ以上残って決まらなければ選ばず review |
| P2 | フォールバックを作らない。カード未検出・向き不定・欄未検出は「処理する」ではなく review。「画像全体をカードとみなす」経路はコードに存在しない |
| P3 | 「塗れた」と「正しい場所を塗った」は別の検証（V1 が必須ゲート） |
| P4 | 向きは 4回転×鏡像 の 8 仮説を全部評価し、一意に決まらなければ review |
| P5 | 出力は PNG のみ・単色塗りのみ（noise・JPEG・ぼかし・モザイク・半透明は実装しない） |
| P6 | 迷ったら review。ok は全ゲートの AND だけ（OR やスコア加算で通さない） |

試作（固定 ROI・面積最大・全体フォールバック・一様性のみの検証）で実測された欠陥は 3 つ: ① 台紙を最大四角形として誤選択（被覆率 0.693）、② 極端な斜めで検出失敗→全体をカードとみなして成功扱い（被覆率 0.53/0.48）、③ verified が位置を見ない。①②③はそれぞれ P1・P2・P3 と、同じ状況を再現するテストで固定している（`tests/test_negative.py`）。

## 1. パイプライン

```
for each input (ソート済みパス順):
  S0 load_and_validate   形式(JPEG/PNG)・50MP 上限・EXIF 向き反映・sha256・長辺 800px 以上
  S1 card_candidates     幾何条件を通った四角形を全部（入れ子含む）返す。0 個なら C_NO_CARD
  S2 hypotheses          候補 × 8 仮説を 1011x638 に正規化し、全テンプレートと ZNCC
  S3 select_unique       best/second のマージン等で一意性を判定。できなければ review
  S4 refine_alignment    ECC(AFFINE) で精密位置合わせ。補正量が大きければ review
  S5 quality_gates       反射・ぼけ・コントラスト・解像度（探索窓内）
  S6 locate_field        T / F / A の3手がかり。一致しなければ review
  S7 build_mask          R = T∪F∪A に余白・保護領域・境界インク・面積妥当性
  S8 apply_and_verify    塗り → V1〜V6（verify.py。別モジュール）
  S9 emit                ok/.staging に masked と preview を書く → 読み戻して V2/V3 → V6 → 2つ揃ってから os.replace
  例外はどの段でも X_EXCEPTION として review（握り潰して ok にしない）
```

各段は「値＋理由コード」を返し、ログの `stages` に全メトリクスを残す。1つでも fail したらその場で review（S8 の検証だけは全項目の値を残す）。
status は `ok` / `review` / `error`（`error` は入力が読めない・出力が既に存在する等。運用上はこれもレビューキュー行き）。

### S0
`np.fromfile` + `cv2.imdecode`（日本語パス対策）。EXIF の向きは **OpenCV 5.0.0 の `imdecode` が反映する**（`IMREAD_IGNORE_ORIENTATION` 付きとの形状比較で実測し、テストで固定。ログ `exif_orientation_applied`）。アルファ付き PNG は白に合成（透過部が黒い縁になるのを避ける）。画素数はヘッダから先に読み、超過なら `E_TOO_LARGE`（デコード前に拒否）。

### S1 カード候補（台紙・紙・机を誤選択しない）
長辺 1200px の作業画像で 3 マップ（Canny(40,120)+3×3膨張 / Otsu / Otsu 反転）→ `findContours`(RETR_LIST, CHAIN_APPROX_NONE) → 凸包 → `approxPolyDP`(eps 0.01〜0.05 を段階的に) で 4 点輪郭。
角は、候補輪郭のうち凸包境界上の点（カード外周）を各辺の近傍帯（対角線長の 1.5%、辺の両端 10% は角丸の弧を避ける）から集め、`fitLine(DIST_HUBER)` した直線の交点に精密化する。

幾何条件（全部 AND。初期値）— 落ちた理由は最初に当たったコード:
面積比 0.15 以上 0.95 未満（`C_AREA`）→ 内角 70〜110°（`C_ANGLE`）→ 対辺長比 ≤ 1.30（`C_PERSPECTIVE`）→ 4 角が画像端から短辺の 0.5% かつ 3px 以上内側（`C_TOUCHES_BORDER`）→ 見かけのアスペクト 1.585±20%（`C_ASPECT`）→ 辺のエッジ支持率（3px 以内に Canny エッジがある割合）≥ 0.85（`C_EDGE_SUPPORT`）→ 同じ四角形（4 角の距離が対角線長の 1%）が 3 マップ中 2 つ以上で見つかる（`C_NO_CONSENSUS`）。
**カードが枠いっぱい**（4 角が画像の 4 隅から 4px 以内、かつ見かけのアスペクトが 1.585±5%）の輪郭は、幾何判定に回す前に除外して印を付ける。他の候補が 1 つも取れなかったときだけ理由コード `C_FRAME_FILLED`（＝余白を残して再撮影）にする。**「画像全体をカードとみなして処理する」経路は無い**。画像全体の輪郭はどの写真にも現れる（Otsu 反転マップの外周）ため、正常にカードが取れている写真では付けない。
通った候補は面積で順位付けせず全部 S2 へ。入れ子（台紙とカード）は両方評価され、台紙は罫線テンプレートと一致しないことで落ちる。候補が 12 個を超えたら `C_TOO_MANY_CANDIDATES`。

### S2 レイアウト署名と照合
正規化画像 → グレー → `adaptiveThreshold(MEAN_C, BINARY_INV, 31, 10)` → 横 41×1・縦 1×25 の OPEN の OR（罫線マップ。文字は消える）→ テンプレートの `ignore_rects` を 0 → INTER_AREA で 101×64 → 弱いガウシアン（σ=1.0、8 章）→ テンプレート署名との ZNCC（分散 0 なら 0）。
テンプレートは `kind = positive`（対象の裏面様式）と `negative`（表面・他様式。一致したら review にする対象外様式）。

### S3 一意性（全部 AND）
`best` = 全仮説の ZNCC 最大、`twin` = `best` の鏡像双子（§1.5c。同じ候補・同じ回転で、正規化画像が左右反転になる仮説。照合相手は `best` と同じ配置ファミリーのテンプレートで、回転番号ではなく四角形の幾何で確かめる）、`second` = `best` と `twin` を除いた最大。
`best` が positive（否なら `O_FRONT_SIDE` / `O_UNSUPPORTED_CARD`）、`best ≥ 0.55`（`O_LOW_SCORE`）、`best − second ≥ 0.10` かつ差が 1e-6 超（`O_AMBIGUOUS`）、非鏡像（`O_MIRRORED`。自動で反転し直さない）、`best − twin ≥ mirror_margin`（`O_MIRROR_AMBIGUOUS`。M2）、別の場所（四角形 IoU < 0.1）に positive ZNCC ≥ 0.55 の候補が無い（`C_MULTIPLE_CARDS`）。同点は並び順に依存させず `O_AMBIGUOUS`。上下逆・時計回りも同じ枠組みで決まる。上下反転（鏡像＋180°）は回転が違うので twin にせず、主マージンの比較対象に残す。
`mirror_margin` は 0 以下にできず、`mirror_m3` は `"unimplemented"` 以外を受け付けない（`ConfigError`。§1.5c の「twin 除外と M2・M3 はセット、無効化は起動時エラー」）。M3 は未実装なので判定を保留し、S3 `mirror.m3` と S6 `M3` に `pending_code: "F_ANCHOR_MIRRORED"` を残す。

### S4 精密位置合わせ
罫線マップ（`ignore_rects` を 0、σ=3 でぼかし）同士で `findTransformECC(MOTION_AFFINE, 50 回, eps 1e-5)`。相関 ≥ 0.70、平行移動 ≤ カード幅の 2%、スケール 0.97〜1.03、回転 ±2° 以内（外れたら `R_ECC_FAIL`）。以後、**元画像 → 位置合わせ済みテンプレート座標系**への写像は「(正規化→元の射影) ∘ (ECC のアフィン)」の合成 `P_t2o` 1 つで一貫させ、1 回の補間で `aligned` 画像を作る（S5〜S7、V1/V4/V5 はこの座標系）。行列は有効数字 12 桁でログに固定（透視項が 1e-7 程度で、小数桁の丸めだと数 px ずれるため）。

### S5 品質（探索窓＝`field_rect` をカード幅・高さの 10% ずつ広げた領域）
3 チャネルとも 250 以上の画素 ≤ 2%（`Q_GLARE`）、Laplacian 分散 ≥ 60（`Q_BLUR`）、p95−p5 ≥ 60（`Q_CONTRAST`）、元画像でのカード長辺 ≥ 800px（`Q_LOW_RES`。S0 の画像長辺とは別に見る）。

### S6 欄の特定（独立した 3 手がかり）
- **T**: テンプレートの `field_rect`（位置合わせ後の座標）。
- **F**: 2 モード。テンプレートが `f_rules`（期待位置の線分）を持つ場合は「線分の一致」で判定する（実様式用。設計書 §4 (D) 第一候補）。横罫線・縦罫線ごとに「期待バンド ±`f_rules_tol_px` に罫線画素が乗っている列（行）の割合」＝支持率を測り、全部が `f_rules_support_min` 以上なら成立（`F_FRAME_NOT_FOUND`）。**閉じた矩形は要求しない**。F の矩形は宣言した期待位置の外接矩形（検出画素の外接にすると 1 本外れた線が他の線の位置を押し広げるため）。`f_rules` を持たないテンプレート（合成）は従来どおり「罫線マップの連結成分のうち外接矩形が T と IoU 最大のもの（下限 0.30）、周長上の罫線支持率 ≥ 0.85」（`F_FRAME_NOT_FOUND`）。`f_rules` は読み込み時に h_lines 3 本以上・v_lines 2 本以上を要求する（F を 1 本の線に弱める書き換えを止める）。
- **A**: 探索窓内で暗画素マップの**外側輪郭**から印の単体候補（`anchor_type = box3`: 4 頂点・辺長が期待値の ±35%・アスペクト 0.80〜1.25・矩形度 ≥ 0.80。`circle3`: 同サイズ・円形度 ≥ 0.80）を取り、3 連条件（辺長の変動係数 ≤ 0.12、x 中心のずれ ≤ 辺長の 0.35 倍、間隔比 ≤ 1.20、間隔が期待値の ±20%）を満たす組が**ちょうど 1 つ**（0 なら `F_ANCHOR_NOT_FOUND`、2 以上なら `F_ANCHOR_MULTIPLE`）。3 連の実位置から（平行移動＋等方スケールの）相似変換を求め、`field_rect` を写したものを A の欄推定とする。手書きで中が埋まっていても外側輪郭で判定する。
- 一致判定: IoU(T,F) ≥ 0.80、IoU(F,A) ≥ 0.85、IoU(T,A) ≥ 0.80、F と A（と 3 つの印）が探索窓内に完全に収まる（`F_CUE_DISAGREE`）。必須の手がかりは `template.required_cues`（既定 T,F,A）。
- 欄矩形 `R` = T ∪ F ∪ A の外接矩形（保守側）。

### S7 マスク矩形（テンプレート座標で作る）
余白は横 `max(12px, 0.06×R幅)`、縦 `max(12px, 0.08×R高)`、各辺の基本余白は 36px 上限。`protected_rects` と重なる辺だけ余白を縮め（保護領域との間に 3px の隙間を空ける）、縮めた後に R から 6px 残らなければ `M_PROTECTED_CONFLICT`。
境界インク: 外周 12px のリングで「暗画素 − テンプレートの印刷物（`print.png` を 5px 膨張）」＝インクを数え、辺ごとにリング内インク比 > 0.5%、またはマスク境界をまたぐ長さ 10px 以上の連結成分があればその辺だけ 12px 広げて再判定（最大 2 回）。解消しない・広げると保護領域に当たる場合は `M_INK_OVERFLOW`。マスク面積がテンプレートの期待値の 0.8〜1.6 倍でなければ `M_AREA_OUT_OF_RANGE`。
元画像座標のマスク多角形 = `P_t2o` で矩形の 4 隅を写したもの。`fillPoly` して 2px 膨張してから塗る。

## 2. 塗り方
既定は黒 (0,0,0)、`--fill white` も可（検証は同じ）。白は「何も書かれていない欄」と見分けがつかず「マスク済み」と判別できないため既定にしない。noise・ぼかし・モザイク・半透明・JPEG 出力は実装しない。再エンコードで EXIF は落ちる（仕様）。

## 3. 検証（`verify.py`。全部 AND、1 つでも落ちたら ok/ に出さない）

入力は原本・出力（配列／ファイル）・ログに記録した幾何（`geometry`: `P_t2o` / `mask_rect` / `mask_polygon` / `cues` / `card_quad`）とテンプレート・設定だけ。マスクを作る側の `mask.py` / `pipeline.py` / `orient.py` は import しない（AST で確認するテストがある）。

| | 判定 | 落ちたとき |
|---|---|---|
| V1 | マスク多角形を正規化座標へ逆写像し、T・F・A（3 つの印の外接矩形）の各隅が内側 6px 以上、`field_rect` の 4 隅が全て内側（100% 包含）、IoU ≥ 0.70、F/A が探索窓内、ログの `mask_rect` と多角形が整合 | `V_INCLUSION` |
| V2 | マスク内が全て塗り色と完全一致（メモリ上と PNG 読み戻し後） | `V_FILL_NOT_UNIFORM` |
| V3 | マスク外が原本のデコード配列と完全一致（読み戻し後。非ゼロ差分 0） | `V_OUTSIDE_CHANGED` |
| V4 | 出力で カード再検出（ログの四角との角のずれ ≤ 対角線長の 1%）、探索窓内の印の単体候補 0・3 連 0、マスクを 3px 侵食した領域の Canny エッジ 0 | `V_CARD_DAMAGED` / `V_REDETECT_TRACE` |
| V5 | 最終マスク矩形の外周リングにインクが残っていない（原本を位置合わせして再計算） | `V_BOUNDARY_INK` |
| V6 | 原本の sha256 と mtime が処理前後で一致 | `X_ORIGINAL_CHANGED`（バッチ停止） |

書き出し: `ok/.staging/` に masked と preview を書き → masked は読み戻して V2/V3、preview はデコード可能なことを確認 → V6・原本不変 → 2つ揃ってから `os.replace` で ok/ へ確定。確定後に失敗したら ok/ に置いた分を削除してから review（削除も失敗したら error＋バッチ停止）。失敗時は staging のファイルを削除。

## 4. ファイル構成

| | |
|---|---|
| `config.toml` | 全しきい値（コードに直書きしない）。欠けたキーは起動時に `ConfigError`（終了コード 1） |
| `templates/<id>/` | `template.json`（`kind` / `field_rect` / `anchor` / `protected_rects` / `ignore_rects` / `required_cues` / `expected_mask_area` …。座標は正規化 1011×638）、`ruling.png`（罫線マップ）、`print.png`（暗画素マップ） |
| `donor_mask/` | `imgio`(入出力) `geom` `layout`(罫線マップ・署名・ZNCC) `template` `card`(S1) `orient`(S2〜S4) `quality`(S5) `field`(S6) `ink` `mask`(S7) `verify`(S8) `pipeline`(統括) `build`(template-build) `cli` |
| `tests/fixtures_gen.py` | 合成データ生成器（固定シード。コミットしない画像を毎回再生成） |
| `tools/build_synthetic_templates.py` | 合成テンプレートの作成（テストが、コミット済みテンプレートとの一致を確認）。実サンプル用は `donor_mask/build.py`（`template-build`） |

ログ JSON は設計書 5.2 のスキーマ（`schema_version`, `job_id`, `input`, `versions`, `config_sha256`, `templates_sha256`, `status`, `reasons`, `stages`, `outputs`, `original_unchanged`, `nondeterministic`）に、検証用の `geometry` を足したもの。`nondeterministic`（開始時刻・処理時間）以外は、同じ入力・同じ設定なら完全一致する。画素・切り抜き・欄の内容は入れない。

## 5. 校正

しきい値・座標は **初期値**で、実物で決める。設計書 5.4 の手順（C0〜C5）に従う:

- C0 合成フィクスチャだけで幾何を固める（v1 はここまで。初期値のまま全テスト通過）。
- C1 テンプレート作成: 本人同意済みの社内サンプル、または公的な見本画像（入手経路と利用可否は要確認）を `rectify` で正規化し、5% グリッド付き画像を見て `field_rect`・アンカーの位置と大きさ・`protected_rects`・`ignore_rects` を記入する。様式ごとに 5 枚以上、`field_rect` は全サンプルの和集合、署名は平均。最初の作業として「実物にチェックボックス3連が本当にあるか（番号を○で囲む方式か）」「欄に枠罫線があるか」を確認し、無ければ `anchor_type` や `required_cues` を変える。
  - `python -m donor_mask template-build --in <正規化済みサンプル> --out templates/<id> --id <id> --spec <測定値.json>`（`donor_mask/build.py`）。1枚目を基準に罫線マップの ECC(AFFINE) で位置合わせし（相関 < `[calibrate] align_max_correlation` は「別様式」としてエラー）、罫線マップ・印刷物マップを平均して `avg_threshold` で二値化する。**座標は推測しない**（`--spec` に無い必須キーはエラー）。
  - サンプルが `[calibrate] min_samples`（5）未満なら**エラー**。実データで 5 枚揃わない場合だけ `--allow-fewer` を明示し、`min_samples_met: false` の**暫定テンプレート**として書き出す（自動 ok の解禁対象にしない）。書くのは `template.json` / `ruling.png` / `print.png` だけで、サンプル画像は複製しない。
  - `anchor_type = "none"` は「選択肢に印刷された印（□3連・○）が実在しない様式」用。このとき `required_cues` に A は入れられない（ローダーが拒否する）。**T だけの `required_cues` は不可**（テンプレート位置だけで塗る経路になるため）。
- C2 シャドーモード（`--shadow`）で本番入力を処理し、全件 review に回して担当者の判定と照合する。
- C3 しきい値: メトリクスごとに陽性・陰性の分布を取り、「以上」型は 陽性の最小値×0.9 と 陰性の最大値+0.02 の大きい方（`calibrate` は未実装。ログの `stages` から手で集計する）。単独で分離できないメトリクスはマージン型へ。
- C4 自動 ok の解禁基準: テンプレートごとにシャドーモードで連続 200 件の誤通過 0 件（0/200 でも 95% 上限は約 1.5%）。それまで自動 ok は無効のまま。
- C5 解禁後も、sha256 を 20 で割った余りが 0 の ok 出力を決定的に抜き取って原本と照合する。
- 誤通過（ok なのに欄の一部が見えていた）が 1 件でも出たら、自動 ok を止めて全件レビューに戻し、原因を校正に反映するまで再開しない。

## 6. 主なしきい値（`config.toml`。初期値）

| 段 | 値 |
|---|---|
| S0 | 画素数 ≤ 50MP、長辺 ≥ 800px |
| S1 | 面積比 [0.15, 0.95)、内角 70〜110°、対辺比 ≤ 1.30、アスペクト 1.585±20%、辺の支持率 ≥ 0.85（3px）、端から ≥ max(3px, 短辺の 0.5%)、合意 ≥ 2 マップ（角 1%） |
| S2/S3 | 署名 101×64（block 31, C 10, 41×1 / 1×25, σ 1.0）、best ≥ 0.55、マージン ≥ 0.10、同点 1e-6、別カード IoU < 0.1 |
| S4 | 相関 ≥ 0.70、平行移動 ≤ 2%W、スケール 0.97〜1.03、回転 ±2° |
| S5 | 白飛び ≤ 2%（≥250）、Laplacian 分散 ≥ 60、p95−p5 ≥ 60 |
| S6 | F 支持率 ≥ 0.85、印の辺 ±35%、アスペクト 0.8〜1.25、矩形度 ≥ 0.8、3 連 CV ≤ 0.12・x ずれ ≤ 0.35・間隔比 ≤ 1.2・期待間隔 ±20%、IoU(T,F) ≥ 0.80・(F,A) ≥ 0.85・(T,A) ≥ 0.80 |
| S7 | 余白 max(12, 6%/8%)・上限 36、保護領域との隙間 3px・残余白 ≥ 6px、面積 0.8〜1.6 倍、リング 12px・インク比 ≤ 0.5%・またぎ ≥ 10px・拡張 12px×最大 2 回 |
| V | 内側距離 ≥ 6px、IoU ≥ 0.70、カード角 ≤ 1%、侵食 3px |

## 7. fail-closed 条件（まとめ）

README の表を参照。コードは `donor_mask/` の各段が理由コードを返す。`ok` に至る道は「S0〜S7 の全ゲート通過 → V1〜V6 通過 → 読み戻し V2/V3 → 原本不変」の 1 本のみ。

## 8. 設計書との差分・未実装・追加（食い違いを避けるための明示）

**未実装（v1 で先送り。呼ぶと終了コード 1 と「未実装」）**
- 設計書 v2.1 §1.5c の **M3（印刷数字「2」「3」の左右反転照合）**。M1（`O_MIRRORED`）・M2（`d_mirror ≥ mirror_margin`）と twin の主マージン除外は実装済み。M3 は未実装なので `[select] mirror_m3 = "unimplemented"` を必須にし（他の値は `ConfigError`）、S3 `mirror.m3` / S6 `M3` に `pending_code: "F_ANCHOR_MIRRORED"`（判定を保留）を残す。**M3 が無い間は、鏡像入力でも d_mirror ≥ 0.03 なら M2 を通過しうる**。
- 設計書 v2.1 §1.5b の「配置ファミリー同士を second にしない」規則は、`field_rect` の IoU ≥ `family_iou` の判定まで実装済みだが、**second からの除外には未適用**（同族テンプレートが 1 つしかない現状では結果に影響しない。回転違いを同族扱いすると向きの曖昧さを検出できなくなるため、定義の詰めが必要）。
- `calibrate`、`make-fixtures`（生成器は `tests/fixtures_gen.py` にあるが CLI 化していない）。`rectify` は実装済みだが、テンプレートに対する最良の非鏡像仮説を直して 5% グリッドを焼くだけ（判定はしない）。`template-build` は実装済み（`donor_mask/build.py`。§5 の C1）。
- `labels.jsonl`・レビュー判定の記録、ok 全件のコンタクトシート目視、`donor-mask` というコマンド名（`python -m donor_mask` で動かす）。
- 実物様式のテンプレートは**暫定版のみ**。同梱は合成（`synthetic_back_v1` positive、`synthetic_front_v1` / `synthetic_other_v1` negative）と、実写真3枚（**同一カードの3撮影**）から作った `real_back_v1`（positive、`n_samples=3` / `min_samples_met: false`）。
  - 実測（正規化 1011×638。罫線込みの和集合）: `field_rect` = x 40〜971 / y 300〜597、`protected_rects` = x 56〜955 / y 52〜298、`ignore_rects` = 宣言ブロック全体。
  - **F（欄の枠罫線）は無い**（長い縦罫線は x≈56 / x≈950 の y≈52〜298 のみ。`detect_frame` が拾えるのはカード外周の枠＝IoU(T,F) 0.43）。**A（印刷された□/○3連）も無い**（等サイズ・等間隔の3連は 0 組）。実在する手がかりは**数字3連 y=362-386 / 392-416 / 422-443（ピッチ 30px）**だけ。
  - したがって `required_cues = ["T","F"]`（`anchor_type: "none"` では A 必須にできず、T だけも不可）となり、S6 は必ず `F_FRAME_NOT_FOUND` か `F_CUE_DISAGREE` で review。**実カードの ok 経路は開かない（fail-closed は維持）**。
  - 実測の停止点: 実写真3枚は既定で S0 `Q_LOW_RES`（554〜696px < 800）、解像度ゲートだけ 200px に下げた診断でも S1 候補 0 件（`C_ASPECT`/`C_PERSPECTIVE`/`C_TOUCHES_BORDER`/`C_ANGLE`/`C_FRAME_FILLED`）。正規化カード＋灰色余白 60px では S2 の ZNCC が 0.794 / 0.567 / 0.557 まで出るが S3 が `O_AMBIGUOUS`（鏡像仮説との差 0.034〜0.087 < 0.10）。S7 は保護領域との隙間が 2px のため `M_PROTECTED_CONFLICT`（`pad_after_shrink=-1.0 < 6`）。合成テンプレートとの ZNCC は +0.005 / +0.022 / +0.069（実テンプレート相手は +0.784 / +0.534 / +0.526）。
  - 未了: 数字3連を新しい `anchor_type` として定義するか、F を「上下罫線ペア」に再定義するか（現行 `detect_frame` は単一連結成分の外接矩形＋周長支持率 0.85 なので、上下だけの枠は支持率 ~0.5 で通らない）。S7 余白規則の実様式対応。様式差の検証（3枚は同一カード）。
  - `templates/real_back_v1/template.json` の全フィールドと根拠（**推測で埋めた値は無い**）:

    | フィールド | 値 | 根拠 |
    |---|---|---|
    | `id` / `kind` / `label` | `real_back_v1` / `positive` / `back` | 裏面の陽性テンプレート。`template-build --id real_back_v1 --kind positive --label back` |
    | `synthetic` | `false` | 生成器の出力ではない（実写真由来）。`build_template` が常に false を書く。テストで明示を固定 |
    | `size` | `[1011, 638]` | 正規化キャンバス（`config.toml` の `canvas`）。入力3枚の切り出しがこのサイズであることを `build_template` が検証 |
    | `window_expand` | `0.1` | 既定値（`field_rect` を 10% 広げた窓で S6 を見る）。実測ではなく運用既定 |
    | `field_rect` | `[40, 300, 971, 597]` | 3枚の罫線込み外接矩形の**和集合**（`cut/field_rect.json`。中央値は x 56〜953 / y 292〜597、ばらつき x±5 / y±2）。和集合にしたのは、正規化誤差で欄が窓から出る側に倒さないため |
    | `anchor_type` / `anchor` | `"none"` / `null` | 実物に印刷された選択肢マーク（□/○3連）が無い（等サイズ・等間隔の3連は 0 組）。推測で `box3` を入れない |
    | `required_cues` | `["T", "F"]` | T（宣言ブロック）は必須。T のみは不可（親レビュー P1-L2）ので F を足す。A は `anchor_type="none"` では必須にできない。**F は実在しない**ので、この組み合わせは「必ず review に落ちる」ことを意味する（fail-closed を維持） |
    | `protected_rects` | `[[56, 52, 955, 298]]` | 上の表（生年月日・住所・交付者など）の外枠。罫線込みの実測。ここは塗ってはいけない |
    | `ignore_rects` | `[[40, 300, 971, 597]]` | 手書き・スタンプが乗りうる宣言ブロック全体。テンプレート平均・署名から除外する（＝筆跡をテンプレートに焼き込まない） |
    | `expected_mask_area` | `345553.56` | 自動導出（`field_rect` を `[mask]` の基本余白で `padded` した面積）。手で書いていない。テストで再計算して固定 |
    | `n_samples` / `min_samples_required` / `min_samples_met` | `3` / `5` / `false` | 入力3枚（同一カードの3撮影）< 設計書 5.4 C1 の5枚以上。`--allow-fewer` で暫定として通し、JSON に明示 |
    | `align_correlations` | `[1.0, 0.8847, 0.9005]` | 平均化前の位置合わせ（ECC）相関。2枚目以降 0.88〜0.90 なので平均像は破綻していない |
    | `note` | 暫定・F 不在の説明 | 上記の結論を JSON 単体で読めるようにした文 |
- 合成テストの変換グリッドは設計書 6.1 の全直積ではなく、一因子ずつ＋複合の固定サンプル。肌色楕円・白飛び楕円・σ=4 のぼけ・枠だけ・3 連が 2 組・はみ出しストロークは実装済み。

**設計書から意図して変えた／補った点**
1. `io.py` は `imgio.py`（標準ライブラリの `io` との衝突を避けるため）。
2. 署名に `blur_sigma = 1.0` の弱いガウシアンを追加（設計書の署名定義にない。細い罫線のわずかなずれに対する ZNCC の頑健化。0 で無効）。
3. S1 の角の精密化は、「辺近傍の Canny エッジ点」ではなく「候補輪郭のうち凸包境界上の点」に `fitLine(DIST_HUBER)` を当てる。内側の罫線・文字のエッジが帯に混ざって直線がずれる実測不具合があったため。
4. V4 のカード再検出は、黒塗りで Otsu の閾値やカード縁のエッジが変わるため、合意マップ数 1・縁の支持率 0.5 に緩め、代わりにログの四角との角の一致 1% を厳密に見る。アンカー検出は S2〜S4 を再実行せず、ログの `P_t2o` で出力を位置合わせして行う。
5. `verify.py` は検出側の部品（`card` / `field` / `ink` / `layout` / `geom` / `imgio` / `template`）を使う（V4・V5 の再検出に必要）。マスクを作る側（`mask` / `pipeline` / `orient`）は import しない。設計書の「マスク処理側のモジュールを import しない」をこう解釈した。
6. 境界インクの拡張は基本余白の上限 36px の外側に最大 12px×2 回まで許す（36px 上限をそのまま適用すると大きな欄では拡張できないため）。
7. S7 に `protected_clear_px = 3`（保護領域との隙間）を追加（元画像側の 2px 膨張が保護領域に食い込まないため）。境界インク判定では正規化キャンバスの縁 6px（カード縁・背景の写り込み）をインクに数えない（`ink.border_ignore_px`）。
8. V1 の「内側 6px」は、元画像上の距離を換算せず、マスク多角形を正規化座標へ逆写像して測る。
9. ECC 前に罫線マップを σ=3 でぼかし、`ignore_rects` を 0 にする（設計書は「罫線マップ同士」とのみ記載）。
10. 件単位の `error`（入力が読めない・既存出力あり）も終了コードは 3（設計書は致命的エラーのみ 1）。
11. 追加の理由コード: `C_AREA` `C_ANGLE` `C_ASPECT` `C_EDGE_SUPPORT`（S1 の個別条件）、`C_TOO_MANY_CANDIDATES`、`F_FRAME_NOT_FOUND`、`V_BOUNDARY_INK`、`X_IO` `X_REVIEW_IMAGE_FAILED`。
12. アルファ付き PNG は白に合成してデコード（透過部が黒くなって縁として誤検出されるのを避ける）。出力は 3 チャネル（アルファは落ちる）。
13. `preview.jpg` / `review.jpg` は JPEG（納品物ではなく目視用。マスク済み画像は PNG のみ）。
14. `--shadow` は実装済み（設計書の v1 先送り対象ではないが安価だったため）。

**校正で決める必要がある値（一覧）**: `field_rect`・アンカー（種類・辺長・間隔・位置）・`protected_rects`・`ignore_rects`・`expected_mask_area`（様式ごと、実サンプル 5 枚以上）、S1 の各しきい値（特に面積比 0.15）、S3 の 0.55 / 0.10、S4 の ECC 許容、S5 のぼけ・コントラスト・解像度、S6 の IoU 群、余白と保護領域の隙間、境界インクの比と膨張量。
