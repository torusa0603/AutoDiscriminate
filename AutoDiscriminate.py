import mWatershed
import sys 
import re
import mEvent
import mSocket
import threading

# Event handler may be a function or a instance method etc.
# Every handler must accept two arguments; a sender and an event-specific 
# parameter.
def event_handle(sender, earg):
    if earg == "go":
        mWatershed.main(False, result_folder_path, False, True)
        c_socket.Send("End!!!")
    elif earg == "end":
        lock.release()


if __name__ == '__main__':
    result_folder_path = str(sys.argv[1])
    c_socket = mSocket.cMain(mSocket.cMain.cMode.tpServer, "")
    c_socket.setDaemon(True)
    c_socket.start()

    # Add event handler
    c_socket.evtRec.evt += event_handle
    c_socket.evtRec.handle_connect = True
    #pub.Action(result_folder_path)
    lock = threading.Lock()
    lock.acquire()
    # ここでロックされる、eventhandle内からロック解除する
    lock.acquire()