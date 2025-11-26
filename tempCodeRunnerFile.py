# Redirect stderr to null at the OS level BEFORE mediapipe loads
# stderr_fd = sys.stderr.fileno()
# null_fd = open(os.devnull, "w")
# os.dup2(null_fd.fileno(), stderr_fd)
