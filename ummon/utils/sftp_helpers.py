import getpass
import os.path as path
import os
import numpy as np
from scipy.stats import describe

HOST, PORT = "141.37.176.188", 22
BASE_URL = "/michael/DoD/baumer_test_prod_var/512x512_samplescale1.0_patchscale1.0/samples"
BASE_DIR = "__sftpcache__"

class SFTP():
    def __init__(self, host=HOST, port=PORT, user=None, password=None, base_dir=BASE_DIR):
        import paramiko
        if path.exists(base_dir) == False: os.makedirs(base_dir)

        # Open a transport
        self.transport = paramiko.Transport((host, port))
        if user is None:
            user = input("Username:")    
        if password is None:
            password = getpass.getpass("Password for " + user + ":") #Prompts for password
        self.transport.connect(username = user, password = password)
        self.sftp = paramiko.SFTPClient.from_transport(self.transport)

    def cd(self, path):
        self.sftp.chdir(path)
        return self

    def get(self, src, dest=BASE_DIR):
        if src[-1] == "*":
            # download directory mode
            local_paths = []
            for f in dir(self):
                local_paths.append(os.path.join(dest, f))
                self.sftp.get(f, local_paths[-1])
            return local_paths
        else:
            self.sftp.get(src, dest)
            return [dest]

    def put(self, src, dest):
        self.sftp.put(src, dest)
    
    def __len__(self):
        return len(self.sftp.listdir())
        
    def __dir__(self):
        return self.sftp.listdir()
        
    def close(self):
        self.sftp.close()
        self.transport.close()