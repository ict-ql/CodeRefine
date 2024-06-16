import os
import subprocess
import time

def goodwait(p):
    while True:
        try:
            rv = p.wait()
            return rv
        except OSError as e:
            if e.errno != errno.EINTR:
                raise

def goodkillpg(pid):
    try:
        if hasattr(os, 'killpg'):
            os.killpg(pid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGKILL)
    except:
        log.error('error killing process %s', pid, exc_info=True)

def call_program(cmd):
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        shell=True)
    try:
        stdout_result = p.stdout.readline()
        while p.returncode is None:
            goodwait(p)
            p.poll()
    except:
        if p.returncode is None:
            goodkillpg(p.pid)
        assert False, "something wrong when run `"+cmd+"`!"

    t1 = time.time()
    return {'success': p.returncode == 0, 'time': t1-t0, 'stdout': stdout_result.decode('utf-8')}

def compile(flag, input_file_path, output_file_path):
    llvm_path="llvm11/llvm/build"
    cmd = llvm_path+"/bin/"+"opt "+flag+" "+input_file_path+" -S -o "+output_file_path
    opt_res = call_program(cmd)
    assert opt_res['success'], "fail to run `"+cmd+"`"

def getAllFilesFromDir(dir):
    file_path_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            path = os.path.join(root, file)
            file_path_list.append(path)
    return file_path_list

def getFileNameFromFullFilePath(full_file_path):
    tmp = full_file_path.split("/")
    file_name_tmpl = tmp[len(tmp)-1].split(".")[:-1]
    file_name = ""
    for fn in file_name_tmpl:
        file_name = file_name +"."+ fn
    return file_name[1:]

def getStrWithoutSpaceSurround(line_str):
    return line_str.lstrip().rstrip()
