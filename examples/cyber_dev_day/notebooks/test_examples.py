cve_code = ["CVE-2022-47011", "CVE-2021-28363", "CVE-2022-47011", "CVE-2023-27043"]
cve_info= ["An issue was discovered function parse_stab_struct_fields in stabs.c in Binutils 2.34 thru 2.38, allows attackers to cause a denial of service due to memory leaks.",
           "The urllib3 library 1.26.x before 1.26.4 for Python omits SSL certificate validation in some cases involving HTTPS to HTTPS proxies. The initial connection to the HTTPS proxy (if an SSLContext isn't given via proxy_config) doesn't verify the hostname of the certificate. This means certificates for different servers that still validate properly with the default urllib3 SSLContext will be silently accepted.",
           "An issue was discovered function parse_stab_struct_fields in stabs.c in Binutils 2.34 thru 2.38, allows attackers to cause a denial of service due to memory leaks",
           "The email module of Python through 3.11.3 incorrectly parses e-mail addresses that contain a special character. The wrong portion of an RFC2822 header is identified as the value of the addr-spec. In some applications, an attacker can bypass a protection mechanism in which application access is granted only after verifying receipt of e-mail to a specific domain (e.g., only @company.example.com addresses may be used for signup). This occurs in email/_parseaddr.py in recent versions of Python"
]

await run_cve_pipeline(pipeline_config, engine_config, [
cve_info])

# The same configuration could be used for testing the microservice example

# for cve in cve_info:
    
#     curl --request POST \
#       --url http://localhost:26302/scan \
#       --header 'Content-Type: application/json' \
#       --data '[{
#           "cve_info" : cve
#        }]'
