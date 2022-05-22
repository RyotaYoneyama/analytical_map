# はじめに
本書は、社内の物体検出の評価基準であるmAPを計算するスクリプトである。特に下記の特徴をもつ。
- mAPを計算する。
- mAPを物体のバウンディングボックス（BB）ごとに計算する．
- TP, FP、FNを6つに分類でき、検出の失敗を分析できる．
- COCOのフォーマットに準ずることで、新たなデータ構造の学習が少ない．
## 背景
物体検出の精度評価としてAverage Presicion (AP) が社内外を含めて広く使われている．
現在はOSSのコードを利用してAPを計算している．
しかしOSSしコードの保守性・拡張性が必ずしも高くなく，追加実装が困難である．
例えば物体の大きさごとに，未検出・ご検出・重複カウントなどを評価したい場合，ドキュメントが少ないOSSのコードを理解して実装するひつようがある．
そこで自社で実装することにより，コードの安全性・保守性・拡張性が高いAPを計算するライブラリの開発を目指す．
## analytical_map基本方針  
* 入力はCOCOフォーマットとする．他フォーマットに関しては変換コードを用いる．
* analytical_map はEvaluate, Analyzeの２段構成となっている．
  * Evaluateでは，バウンディングボックスを分類する．
    * eval:バウンディングボックスの分類はまず{'TP', 'FP'，'FN'}のカウント分類を行い，さらにそれらカウントを｛'Match', 'DC', 'LC', 'Cls', 'Loc', 'Bkg', 'Miss'｝のタイプに分類する．
    * 上記分類を中間ファイル（Middle file）として出力する．
  * Analizeでは，上記中間ファイルを入力として，タイプごとのAP，Precision,Recallを計算し出力する．
    * ap_analyzer：APをカテゴリ数ｘBBサイズで出力する．
    * precision_analyzer：Precisionをカテゴリ数出力する．
    * recall_analyzer：Recallをカテゴリ数出力する．
    * visualize:｛'Match', 'DC', 'LC', 'Cls', 'Loc', 'Bkg', 'Miss'｝を可視化する．
 
# 参考資料
* [Use flow chart](docs/figures/use_flow.drawio.png) 
* [API](https://ryotayoneyama.github.io/analytical_map/)
 
# ソースコード
ディレクトリ階層を下記する。  
analytical_map  
├analytical_map : ソースコード
├debug : デバッグ用ツールおよび描画結果
├docs : sphinx
├docker : Dockerfile  
├sample_data : サンプル用データ
├sample_results　:　サンプル用データに対する出力
└README.md :   


# 基本設計
## クラス図概観
ym_cameraはカメラクラスだけでなく、デモアプリケーションも含めている。  
各クラスの概観を下図に示す。  
![クラス図概観](./ym_cameraクラス図概観.png "クラス図概観") 
### YmCameraクラス
* YmCameraクラス (汎用クラス) をカメラ別の専用クラス (FLIRCameraクラスなど) のスーパークラスとしている。
* 単眼カメラ、ステレオカメラ、Depthカメラからの画像取得の際、カメラ専用クラスを直接呼び出すのではなく、YmCameraクラスを通して行う。
* カメラの機種情報は、ソースコード内の列挙型に記入する。
### YmAppクラス
* YmAppクラスでは、カメラに依存しないアプリケーション用処理を定義する。  
  例えば、YmCameraDemoクラスはYmAppを継承し、YmCameraの操作に関わるデモ処理を実装している。
### CameraEventクラス
* CameraEventクラスは、コールバックを扱うクラスである。  
* カメラSDKによっては、カメラ内部で発生するイベント (画像取得完了, 露光完了等) に対するコールバックを設定できる。  
  CameraEventクラスはコールバックに対する共通処理を管理する。  
  * 現状 (2021/08/20時点)   
  FLIRCameraの場合は、コールバック時の処理にカメラ依存が強く、CameraEventクラス内に処理を記載するのが難しいため、
  FLIRCameraのコールバック処理の実装については、リファクタリング前の状態を維持する。  
  他メーカーのカメラについてもCameraEventの利用はない。
* CameraEventクラス内には、YmCameraクラスのポインタをメンバ変数に用意する。  
  CameraEventクラスのコンストラクタで、YmCameraクラスのポインタを受け取り、メンバ変数に格納する。  
* YmCameraクラス内に、コールバック関数登録用の関数 registerEventCallback を新設する。  
  CameraEventクラス内でコールバック時に呼び出される関数 (処理) をあらかじめ記載しておき、  
  YmCameraクラスのregisterEventCallback 関数を用い、コールバック関数を登録する。  
  なお、CameraEventクラス内の各コールバック用関数は、void *型の引数を持ち、どのような引数でも受け取れるようにしておく。  
### IDepthCameraインターフェース
* Depthカメラを操作するためのインターフェース。
* Depthカメラはこのインターフェースを介して、Depth画像の取得などを行える。
* YmCameraクラスとIDepthCameraインターフェースを多重継承してDepthCameraクラスを定義することを想定している。
* YmCameraを継承した専用カメラクラス（子クラス）がある場合は、子クラスとIDepthCameraインターフェースを継承してDepthCameraクラスを定義することをしている。
### 設定ファイル
* カメラやアプリケーションの設定はハードコーディングしない。  
* 外部から変更可能とするため、「YAML形式」の設定ファイルを読み出して設定を反映する方式とする。  
* カメラ用設定ファイル（例. config_camera_flir_1.yml）とアプリケーション用設定ファイル（例. config_app.yml）がある。  
* 複数カメラに対応する。利用するカメラ用設定ファイルはアプリケーション用設定ファイルにパスを記入することで指定可能である。  
* 設定ファイルの詳細については後述する。  
# 詳細設計
doxygenで生成したドキュメントを参照のこと。  
他に参照可能なドキュメントや各クラス備考について下記する。
## FLIRCameraクラス
画像取得ソフトウェア設計仕様書.xlsx　を参照のこと。  
* YmCameraクラスからの拡張機能  
  * シーケンス＆バースト画像取得機能（カメラ依存）  
    複数露光時間とバースト画像取得枚数を指定することで  
    露光時間をシーケンシャルに変えながら所定の枚数画像を取得することができる。  
![機能BurstSequence](./機能_BurstSequence.jpg "クラス図概観") 
  * SDKを利用した画像保存に対応   
## LUCIDCameraクラス
カメラ制御ソフトウェア開発_LUCID_設計仕様書.xlsx　を参照のこと。
* YmCameraクラスからの拡張機能  
  * HDRカメラ対応  
    HDRカメラ向けの設定として、メイン・サブピクセルの露光時間とゲインを設定できる。  
## BASLERCameraクラス
2021/09/09時点で、Basler pylon SDK 6.2.0.18677とLUCID Arena SDK 1.0.29.5は名前空間で競合しており、どちらか一方しか利用できない。  
メーカーに問い合わせたが、解決方法が無いとのこと。
## ITDLabCameraクラス
2021/09/09時点で、ITDLabはステレオカメラしか供給していないため、ITDLabCameraクラスは実装していない。  
よって、ITDLabCameraStereoクラスのみ実装している。  
# 機能拡張について
## 1. 機能拡張時の方針
機能拡張時の大まかな方針を以下に述べる。  
### ①カメラ種類の登録
カメラ種類は、列挙体 eCameraInfo に登録する。  
### ②カメラ操作用の関数の記述
カメラ操作用の関数は、YmCameraクラスの継承クラスに記述する。  
(FLIRカメラ用の場合は、FLIRCameraクラスが該当)  
* YmCameraクラスに用意されている関数群に対しては、必要に応じて、継承クラス側でオーバーライドする。  
* カメラ固有の処理については、継承クラスで関数を用意し、実装する。
### ③YmAppクラスの継承クラスの追加、main関数の作成  
YmAppクラスの継承クラスを新規に追加し、main関数・アプリ処理を記述
## 2. 機能拡張時の詳細手順
機能拡張時の詳細手順を以下に述べる。
### ①カメラ種類の追加  
ym_camera.h 内に、列挙体 eCameraInfo が定義されている。  
カメラ種類を追加したい場合は、eCameraInfo の列挙子を追加する。  
例) 新しいカメラ種類を「CameraA」という名称で追加したい場合  
|追加前の列挙体 | |追加後の列挙体 |
|:--|:--|:--|
|enum eCameraInfo {<br>　　　　FLIR,<br>};  |  →   | enum eCameraInfo {<br>　　　　FLIR,<br>　　　　CameraA,<br>};  |
### ②カメラ操作用の関数の記述
YmCameraクラスに用意されている関数のうち、継承クラスで実装が必要なもの (下表) を実装する。  
**表. YmCameraクラスに用意されている関数のうち、継承クラスで実装が必要なもの**
|関数名										|関数の説明 |
|:--|:--|
|open()										|カメラの起動時の処理を記述する関数。 |
|close()									|カメラの終了時の処理を記述する関数。 |
|init()										|カメラの初期化時の処理を記述する関数。 |
|start()									|カメラのストリーミング（画像取得）開始処理を記述する関数。 |
|start(struct StreamConfig &cfg)			|カメラのストリーミング（画像取得）開始処理を記述する関数。<br>Burst Sequenceによる画像取得などストリーミングのConfigurationを指定する |
|stop()										|カメラのストリーミング（画像取得）停止処理を記述する関数。 |
|getImage(cv\::Mat &image)					|カメラより得られた画像をcv\::Mat形式で返す関数。 |
|getImage(cv\::Mat &image, int trigger)		|ソフトウェアトリガーを使用し得られた画像をcv\::Mat形式で返す関数。<br>trigger(int)→getImage(cv::Mat)を一つにまとめた関数。 |
|getImage(std\::vector\<cv::Mat> &image, int trigger)		|ソフトウェアトリガーを使用し得られた複数画像をcv\::Mat形式で返す関数。<br>(2021/12/08時点) 本関数で複数画像を取得するためにはstart(struct StreamConfig)でBurst処理を有効にしておく必要がある |
|trigger(int command)						|ソフトウェアトリガー発行処理を記述する関数。 |
|setParams(enum type, std\::string value)<br>setParams(enum type, int value)<br>setParams(enum type, double value) |列挙体 eCameraParameter で定義されているパラメータに値をセットする関数。<br>第1引数に eCameraParameter の列挙子を、<br>第2引数に セットしたいパラメータの値を指定する。<br>第2引数の型 (string / int / double / bool) に応じて、4種類の関数のオーバーロードが必要。|
|setParams(enum type, std\::vector\<int> &value)<br>setParams(enum type, std\::vector\<double> value) |列挙体 eCameraParameter で定義されているパラメータに値をセットする関数。<br>第1引数に eCameraParameter の列挙子を、<br>第2引数に セットしたいパラメータのstd::vectorの参照を指定する。<br>第2引数の型 (int / double) に応じて、関数のオーバーロードが必要。<br>カメラがSequencer機能などに対応している場合に利用する|
|getParams(enum type, std\::string &value)<br>getParams(enum type, int &value)<br>getParams(enum type, double &value) |列挙体 eCameraParameter で定義されているパラメータの値を取得する関数。<br>第1引数に eCameraParameter の列挙子を、第2引数に値を保存したい変数を参照渡しで指定する。第2引数の型 (string / int / double / bool) に応じて、4種類の関数のオーバーライドが必要。|
|getParamsRange(enum type, int *maxval, int *minval)<br>getParamsRange(enum type, double *maxval, double *minval)<br> |列挙体 eCameraParameter で定義されているパラメータの設定範囲を取得する関数。<br>第1引数に eCameraParameter の列挙子を、第2, 3引数に設定範囲を保存したい変数を参照渡しで指定する。第2, 3引数の型 (int / double) に応じて、3種類の関数のオーバーライドが必要。|
|getParamsIncVal(enum type, int &value)<br>getParamsIncVal(enum type, double &value) |列挙体 eCameraParameter で定義されているパラメータの設定ステップを取得する関数。<br>第1引数に eCameraParameter の列挙子を、第2引数に設定ステップを保存したい変数を参照渡しで指定する。第23引数の型 (int / double) に応じて、3種類の関数のオーバーライドが必要。|
|onReadConfigFile(YAML::Node yaml)			| カメラ用設定ファイルのうち、カメラ固有の設定値を読み込む関数。設定ファイルでの設定項目は、カメラごとに自由に設定可。|
### ③YmAppクラスの継承クラスの追加、main関数の作成
YmAppクラスの継承クラス (FLIRカメラの場合は、YmCameraDemoクラスが該当) を用意する。必要に応じて、追加したいアプリ関数を本クラス上に追加する。  
また、onReadSettingsFile(YAML::Node yaml) 関数をYmAppクラスからオーバーロードし、アプリの設定ファイルの読み込み処理を記述する。
main関数では、以下の処理を記述する。
1) YmAppの継承クラスのインスタンスを作成
2) YmAppクラスのgetInstance関数を呼び出し、カメラ操作用クラスのインスタンスを作成
3) YmAppクラスのloadSettingsApp関数を呼び出し、アプリの設定ファイルを読み込む
4) アプリ処理を記述する (必要に応じて、同クラス内にアプリ関数を記載し、呼び出す)。
# 設定ファイル
設定ファイルの実装概要と、設定ファイルの記述ルールについて述べる。  
実装詳細は、doxygenで生成したドキュメントもしくはソースコードを参照のこと。  
なお、設定ファイルはカメラ用とアプリケーション用の2種類がある。  
## カメラ設定ファイル
* YmCamera\::readConfigFile(std\::string filepath)
  全カメラ共通で使えるパラメータの読み出し処理を実装する。  
  また、YmCameraを継承したクラスに実装された onReadSettingFile(YAML\::Node yaml) を呼び出す。  
  ソースコードは ym_camera.cpp  
* カメラ専用クラスのメンバ関数 onReadSettingFile(YAML\::Node yaml)
  各メーカーのカメラに特化したパラメータの読み出し処理を実装する。  
  ソースコードは各カメラクラスに記述する。(例. FLIRカメラクラス FLIRCamera ・・・ flir_camera.h)  
* **設定についての説明は、ソースコードにコメント記述すること**
## アプリケーション設定ファイル
* ConfigAppBaseクラスにアプリケーションに依存しない共通のパラメータを定義する。  
  ソースコードは cfg_app_base.h  
* ConfigAppクラス (ConfigAppBaseクラスを継承) に YmApp の基本的なパラメータを定義する。  
* YmApp\::parse_yaml(std::string filepath) にパラメータの読み出し処理を実装する。 
* アプリケーションクラス(YmAppを継承)のonReadSettingFile(YAML::Node yaml)にアプリケーションに特化したパラメータ読み出し処理を実装する。  
* **設定についての説明は、ソースコードにコメント記述すること**
## 設定ファイル記述ルール
設定ファイルの記述方法はYAMLフォーマットに即し、　[プロパティ]: [値]　という形式で指定する。  
　例) fileoutpath: ./out/  
メンバ変数の型(内部)が構造体(struct)の場合は、[プロパティ]:　と記載後に改行を行い、  
次行以降に　[サブプロパティ]: [値]　という形式で指定する (設定値が複数ある場合は、左記の指定を必要行数分だけ行う)。  
　例) logging :  
  　　　　file : true  
  　　　　console : true  
また、std::vectorとして定義されたプロパティに関しては、任意行数分だけ記載することでリスト化可能。  
記載したい行数分だけ、　- [設定値]　という形式で指定する。  
　例）camera_config:  
  　　　-  
  　　　　caminfo: 0  
  　　　　filepath: .\\config_camera_flir_1.yml  
  　　　-  
  　　　　caminfo: 0  
  　　　　filepath: .\\config_camera_flir_2.yml  
  　　　-  
  　　　　caminfo: 0  
  　　　　filepath: .\\config_camera_flir_3.yml  
# 開発環境
## Windows
* Windows10 x64  
* VisualStudio2017  
* OpenCV 3.4.10 (CUDA利用時は3.4.16)  
  自環境に合わせてビルド～インストールする。  
  opencv_contrib も必要。  
  cmakeデフォルトでビルドした[リンク ](https://ymr-rdu-redmine.com/dmsf/files/6608/view)の7zファイルを解凍して C: 直下に置いても良い。(CUDAなどは使えない)  
* boost 1.69    
  先進的なC\++機能やC++のPythonラッパーを実装するために利用する。
  [リンク ](https://ymr-rdu-redmine.com/dmsf/files/6604/view)の7zファイルを解凍して C: 直下に置く
* yaml-cpp  
  YAML形式の設定ファイルを読み出すために使う
  [リンク ](https://ymr-rdu-redmine.com/dmsf/files/6605/view)の7zファイルを解凍してを C: 直下に置く。
* カメラメーカーのSDK  
  各メーカーのウェブサイトからダウンロードするか、[リンク ](https://ymr-rdu-redmine.com/projects/recognition2016/dmsf?folder_id=1376)のインストーラを使う。
  * FLIR SpinnakerSDK 2.4.0.144
  * LUCID ArenaSDK 1.0.29.5
  * Basler pylon SDK 6.2.0.18677  
    * **※LUCID ArenaSDKと名前空間で競合しているためLUCIDとの同時利用は不可**
    * TOFカメラ blaze101を使う場合はpylon_Supplementary_Package_for_blazeもインストールする
  * ITDLab 納品データ https://ymr-rdu-redmine.com/dmsf/files/6609/view
    * ITDLAB_納品データ/ITDLAB を任意のディレクトリにコピーする。
    * ITDLAB_納品データ/FTD3XXDriver_WHQLCertified_v1.3.0.4 のUSBドライバを登録する。
## Linux
* Ubuntu18.04LTS  
* OpenCV 3.4.10  
* boost 1.65 (ubuntuプリインストール)  
* yaml-cpp  
* カメラメーカーのSDK  
  * FLIR SpinnakerSDK 2.4.0.144  
  * LUCID ArenaSDK 1.0.29.5  
  * Basler pylon SDK　 **※LINUXでの動作は未確認**  
  * ITDLab　**※LINUXでの動作は未確認。メーカーから開発環境を入手する必要がある**  
* インストール手順  
  下記リンクの手順はARMアーキテクチャ向けだがx64に読み換え可能  
  https://ymr-rdu-redmine.com/projects/smss-_sec/wiki/Jetson_NX_Xavier_SetUp
# ROS
Linux環境向けにROSノードも実装している。  
使用方法は https://ymr-rdu-redmine.com/dmsf/files/6577/view を参照のこと。  