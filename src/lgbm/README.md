# LGBMのモデル

## 使い方
`main.py`でpreprocessorを初期化する。  
初期化時にjoblibでダンプしたのがあるならそのパスを指定する。  
特徴量を追加するには、まず追加したい特徴量とitem_idが入ったファイルを作ってそれを読みこむ
preprocessorのadd_featureメソッドを呼ぶといい感じに追加してくれる。  
add_featureの引数は、トレーニングセットの特徴量を読み込んだものと、テストセットの特徴量を読み込んだものと、追加するカラム名のプレフィクス。プレフィクスに関してはオプションになってる

まぁ使いたかったら使ってわからんかったら使わなくていいという感じ
パスが僕は玉木と違う構造になってるのでそのへんだけ注意が必要。
ディレクトリ構造を詳しく教えてくれたら治せるよ〜