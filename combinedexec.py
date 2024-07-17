from recognition import Recognition
from visualization import BeamVisualizer
from multiprocessing import Process, Pipe

def start_recog(pipe_output):
    recog = Recognition(pipe_output)
    recog.start_process()
    pipe_output.close()

def start_vis(pipe_input):
    vis = BeamVisualizer(pipe_input)
    vis.start_process()
    pipe_input.close()
    
if __name__ == "__main__":
    pipe_input, pipe_output = Pipe()
    # pipe_input.close()
    # vis = BeamVisualizer(pipe_input)
    # recog = Recognition(pipe_output)
    # p = Process(target=vis.start_process)
    # q = Process(target=recog.start_process)
    p = Process(target=start_recog, args=(pipe_output,))
    q = Process(target=start_vis, args=(pipe_input,))
    p.start()
    q.start()
    p.join()
    q.join()
