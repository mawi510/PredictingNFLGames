import paramiko
import os

def run_remote_command(hostname, username, pem_file_path, command):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # Load the private key from the .pem file
        key = paramiko.RSAKey.from_private_key_file(pem_file_path)
        
        ssh.connect(hostname, username=username, pkey=key)
        stdin, stdout, stderr = ssh.exec_command(command)
        print(stdout.read().decode())
        print(stderr.read().decode())
    finally:
        ssh.close()

if __name__ == "__main__":
    hostname = os.getenv("EC2_PUBLIC_IP")
    username = os.getenv("EC2_USERNAME")
    pem_file_path = "key.pem"
    command = "sudo certbot renew --nginx --cert-name promatchpredict.com"
    
    run_remote_command(hostname, username, pem_file_path, command)
