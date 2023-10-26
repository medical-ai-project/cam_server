## grad-cam生成プログラム
### 説明
outputディレクトリに、camのヒートマップ画像とその圧縮ファイル(cam_images.zip)が生成される。  
胸部x線画像をhttpリクエストで送信すると、圧縮ファイル(cam_images.zip)がhttpでレスポンスされる。
### 準備
google driveで共有済みのモデル("/グループ1/ai_notebook等/models/efficientNet_xray4.h5")  
をルートディレクトリの中(medic/efficientNet_xray4.h5)に配置してください。  



## ２つの起動方法

### uvicornで起動する方法
appディレクトリ内でuvicornを実行。ポート番号等は各自必要に応じて指定。
```
# exsample. 
uvicorn server:app --reload
```

### docker imageを使用して起動する方法
※cpuが弱いとフリーズすることがあるから注意が必要。ノートpc非推奨。
https://hub.docker.com/repository/docker/ssshelloworld/cam_fastapi/general
```
docker pull ssshelloworld/cam_fastapi
docker run -d --name my_container -p 8000:8000 ssshelloworld/cam_fastapi
```

