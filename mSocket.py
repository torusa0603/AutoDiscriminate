import socket
import threading
from enum import Enum
from abc import ABCMeta, abstractmethod, ABC
import mEvent


# イベントハンドラ
class cEventHandle(object):
    # Socketというタブ名で登録
    m_evt = mEvent.Event('Socket')
    # イベントを登録しているかを示す、これをTrueにしないとソケット内でのイベントは発生しない
    m_bHandleConnect = False
    
    # イベントを発生させる
    def Action(self, arg):
        # Do some actions and fire event.
        self.m_evt(arg)

# 定義クラス
class DefineValue():
    PORT_NUMBER=50000
    BUFFER_SIZE = 16
    END_COMMAND = "Quit"


# メイン関数
class cMain(threading.Thread):
    class cMode(Enum):
        tpServer = 0
        tpClient = 1

    _meMode = cMode.tpServer

    # コンストラクタ
    def __init__(self, neMode, nstrAdress, niPort=DefineValue.PORT_NUMBER):
        # イベントハンドラの作成
        self.evtRec = cEventHandle()
        # モード(サーバー・クライアント)を選択
        self._meMode = neMode
        # モードに応じてインスタンスを変える
        if self._meMode == self.cMode.tpServer:
            # 自動でIPアドレスは取得する
            self._Socket = cSocketAsServer(nstrAdress, self.evtRec, niPort)
        else:
            # アドレスに"host"と入力されると、自動でIPアドレスを取得する
            self._Socket = cSocketAsClient(nstrAdress, self.evtRec, niPort)
        # スレッドクラスのコンストラクタ実施
        super(cMain, self).__init__()
    
    # 通信開始メソッド
    def run(self):
        self._Socket.Connect()

    # コマンド送信
    def Send(self, Command):
        self._Socket.Send(Command)


# ソケットの親クラス(抽象)
class AbstractSocket(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self,Address,Publisher, niPort):
        pass
    
    @abstractmethod
    def Connect(self, ncInstance):
        ncInstance._m_bOpend = False
    
    @abstractmethod
    def Send(self, n_strCommand):
        pass


# サーバーモードのソケットクラス
class cSocketAsServer(AbstractSocket):
    def __init__(self,Address,ncEventHandle, niPort):
        # メンバー変数
        self._m_iPortNumber= niPort
        self._m_cEventHandle=ncEventHandle

    def Connect(self):
        super().Connect(self)
        #ソケットを作成
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as self._m_Server:
            # IPアドレスとポート設定
            # ホスト名を取得する
            _str_host= socket.gethostname()
            # バインドする
            self._m_Server.bind((socket.gethostbyname(_str_host), self._m_iPortNumber))
            self._m_Server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._m_Server.listen(1)
            # 通信を開始する
            while True:
                # 接続待ち状態
                print("Waitting...")
                self._m_clientsocket, _address = self._m_Server.accept()
                print(f"Connecting_{_address}")
                # 通信状態フラグを上げる
                self.m_bOpend = True
                while True:
                    try:
                        # 受信コマンドを待つ
                        _str_rec = self._m_clientsocket.recv(DefineValue.BUFFER_SIZE)
                        # 受信データをデコード
                        _bi_recmsg = _str_rec.decode('utf-8')
                        print(_bi_recmsg)
                        # 無事受信したと返す
                        _str_send_msg = f"OK_{_bi_recmsg}"
                        self.Send(_str_send_msg)
                        # イベントハンドラが登録されていればイベントを発生させる
                        if self._m_cEventHandle.handle_connect == True:
                            self._m_cEventHandle.Action(_bi_recmsg)
                    except:
                        break
                    # 終了コマンドが来たら終了し、接続待ち状態になる
                    if _bi_recmsg == DefineValue.END_COMMAND:
                        break
                # 通信状態フラグを下げる
                self.m_bOpend = False
                self._m_clientsocket.close()
                print("End_Server")

    # コマンド送信
    def Send(self, n_strCommand):
        # 通信を可能な時のみ行う
        if self.m_bOpend:
            # コマンドをエンコード
            _str_send_msg_binary = n_strCommand.encode('utf-8')
            # コマンド送信
            self._m_clientsocket.send(_str_send_msg_binary)
            print(n_strCommand)
            return 0
        else:
            return -1


# クライアントモードのソケットクラス
class cSocketAsClient(AbstractSocket):
    _m_strAdress = ""

    def __init__(self,nstrAddress,ncEventHansle, niPort):
        self._m_strAdress = nstrAddress
        self._m_iPortNumber= niPort
        self._m_cEventHadle=ncEventHansle

    def Connect(self):
        super().Connect(self)
        #ソケットを作成
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as self._m_Client:
            # ホスト接続しする場合
            if self._m_strAdress == "host":
                _str_host= socket.gethostname()
                self._m_strAdress = socket.gethostbyname(_str_host)
            self._m_Client.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # 接続
            self._m_Client.connect((self._m_strAdress, self._m_iPortNumber))
            # 通信を開始する
            # 通信状態フラグを上げる
            self.m_bOpend = True
            while True:
                try:
                    # 受信コマンドを待つ
                    _bi_rec = self._m_Client.recv(DefineValue.BUFFER_SIZE)
                    # 受信データをデコード
                    _str_recmsg = _bi_rec.decode('utf-8')
                    # イベントハンドラが登録されていればイベントを発生させる
                    if self._m_cEventHadle.handle_connect == True:
                        self._m_cEventHadle.Action(_str_recmsg)
                    print(_str_recmsg)
                except:
                    break
                # 終了コマンドが来たら終了
                if _str_recmsg == DefineValue.END_COMMAND:
                    break
            # 通信状態フラグを下げる
            self.m_bOpend = False
            self._m_Client.close()
            print("End_Client")
    
    # コマンド送信
    def Send(self, n_strCommand):
        # 通信を可能な時のみ行う
        if self.m_bOpend:
            # コマンドをエンコード
            sendmsg = n_strCommand.encode('utf-8')
            # コマンド送信
            self._m_Client.send(sendmsg)
            print(sendmsg)
            if n_strCommand == "end":
                self._m_Client.close()
            return 0
        else:
            return -1


# このファイルがメインとして呼び込まれた場合の処理("python mSocket.py"とコマンドから起動)
if __name__ == '__main__':
    # ソケットタイプを選択
    str_connect_type = input("Server? Client? : ")
    if  str_connect_type == "S":
        # サーバーとして作成
        c_socket = cMain(cMain.cMode.tpServer, "")
    elif str_connect_type == "C" :
        # IPアドレスを入力
        str_adress = input("Adress : ")
        # クライアントとして作成
        c_socket = cMain(cMain.cMode.tpClient, str_adress)
    else:
        # 終了
        quit
        
    # メインプログラムが終了した時にスレッドも一緒に終了する設定
    c_socket.setDaemon(True)
    # c_socket内のrun()メソッドが起動する
    c_socket.start()

    # 入力待ち状態をループさせる
    while True:
        str_key = input("key :")
        if str_key == "quit":
            # 終了コマンドを送信し、プログラムを終了させる
            c_socket.Send(DefineValue.END_COMMAND)
            break
        else:
            # コマンドを送信する
            c_socket.Send(str_key)
        if (str_connect_type == "C") & (str_key == DefineValue.END_COMMAND):
            break