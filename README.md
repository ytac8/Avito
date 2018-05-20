# Avito
kaggle Avito Competition


## todo
1. LGBMの分類機をまず自分で作ってみる
2. CharLevelCNNとか使ってtitleとかdescriptionとかをベクトル化
3. ユーザごとの系列データとしてLSTMを回して予測するモデルを作る
4. 画像だけを特徴に分類機を作ってみる
5. ImageNetを使って画像分類し、そのラベルとタイトルデータの距離を特徴量に入れてみる
6. desctioptionの特徴だけで分類してみる


## model
モデルの方針としては３つくらい
- フル深層学習のモデル
- LGBMのモデル
- 深層学習によって得られた特徴をLGBMに食わせるモデル
