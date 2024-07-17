from recognition import Recognition
from visualization import BeamVisualizer
from multiprocess import Process, Pipe, Queue

def start_recog(queue):
    recog = Recognition(queue)
    recog.start_process()
    # pipe_output.close()

def start_vis(queue):
    vis = BeamVisualizer(queue)
    vis.start_process()
    # pipe_input.close()
    
if __name__ == "__main__":
    # pipe_input, pipe_output = Pipe()
    q = Queue()
    a = Process(target=start_recog, args=(q,))
    b = Process(target=start_vis, args=(q,))
    a.start()
    b.start()
    a.join()
    b.join()
