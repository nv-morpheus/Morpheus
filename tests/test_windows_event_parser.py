# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import cudf

from morpheus.parsers.windows_event_parser import WindowsEventParser

TEST_DATA = [
    '{"_indextime":"1554145632","linecount":"63","sourcetype":"WinEventLog:Security","_cd":"309:1061724899","_raw":"04/01/2019 07:07:21 PM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4624\\nEventType=0\\nType=Information\\nComputerName=test109.test.com\\nTaskCategory=Logon\\nOpCode=Info\\nRecordNumber=13730612955\\nKeywords=Audit Success\\nMessage=An account was successfully logged on.\\r\\n\\r\\nSubject:\\r\\n\\tSecurity ID:\\t\\tNULL SID\\r\\n\\tAccount Name:\\t\\t-\\r\\n\\tAccount Domain:\\t\\t-\\r\\n\\tLogon ID:\\t\\t0x0\\r\\n\\r\\nLogon Type:\\t\\t\\t3\\r\\n\\r\\nImpersonation Level:\\t\\tImpersonation\\r\\n\\r\\nNew Logon:\\r\\n\\tSecurity ID:\\t\\ttest.com\\test106$\\r\\n\\tAccount Name:\\t\\ttest106$\\r\\n\\tAccount Domain:\\t\\ttest.com\\r\\n\\tLogon ID:\\t\\t0x9DE8990DE\\r\\n\\tLogon GUID:\\t\\t{E53069F0-662E-0C65-F889-AA8D8770D56A}\\r\\n\\r\\nProcess Information:\\r\\n\\tProcess ID:\\t\\t0x0\\r\\n\\tProcess Name:\\t\\t-\\r\\n\\r\\nNetwork Information:\\r\\n\\tWorkstation Name:\\t\\r\\n\\tSource Network Address:\\t100.00.100.1\\r\\n\\tSource Port:\\t\\t39028\\r\\n\\r\\nDetailed Authentication Information:\\r\\n\\tLogon Process:\\t\\tKerberos\\r\\n\\tAuthentication Package:\\tKerberos\\r\\n\\tTransited Services:\\t-\\r\\n\\tPackage Name (NTLM only):\\t-\\r\\n\\tKey Length:\\t\\t0\\r\\n\\r\\nThis event is generated when a logon session is created. It is generated on the computer that was accessed.\\r\\n\\r\\nThe subject fields indicate the account on the local system which requested the logon. This is most commonly a service such as the Server service, or a local process such as Winlogon.exe or Services.exe.\\r\\n\\r\\nThe logon type field indicates the kind of logon that occurred. The most common types are 2 (interactive) and 3 (network).\\r\\n\\r\\nThe New Logon fields indicate the account for whom the new logon was created, i.e. the account that was logged on.\\r\\n\\r\\nThe network fields indicate where a remote logon request originated. Workstation name is not always available and may be left blank in some cases.\\r\\n\\r\\nThe impersonation level field indicates the extent to which a process in the logon session can impersonate.\\r\\n\\r\\nThe authentication information fields provide detailed information about this specific logon request.\\r\\n\\t- Logon GUID is a unique identifier that can be used to correlate this event with a KDC event.\\r\\n\\t- Transited services indicate which intermediate services have participated in this logon request.\\r\\n\\t- Package name indicates which sub-protocol was used among the NTLM protocols.\\r\\n\\t- Key length indicates the length of the generated session key. This will be 0 if no session key was requested.","_pre_msg":"04/01/2019 07:07:21 PM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4624\\nEventType=0\\nType=Information\\nComputerName=test109.test.com\\nTaskCategory=Logon\\nOpCode=Info\\nRecordNumber=13730612955\\nKeywords=Audit Success","splunk_server":"sc.lab.test.com","source":"WinEventLog:Security","host":"test109","_serial":"5613","_bkt":"wineventlog~309~8C261931-2C10-4450-B82C-39A63512E150","_sourcetype":"WinEventLog:Security","EventCode":"4624","index":"wineventlog","_si":["sc.lab.test.com","wineventlog"],"_time":"1554145641","Message":"An account was successfully logged on.\\r\\n\\r\\nSubject:\\r\\n\\tSecurity ID:\\t\\tNULL SID\\r\\n\\tAccount Name:\\t\\t-\\r\\n\\tAccount Domain:\\t\\t-\\r\\n\\tLogon ID:\\t\\t0x0\\r\\n\\r\\nLogon Type:\\t\\t\\t3\\r\\n\\r\\nImpersonation Level:\\t\\tImpersonation\\r\\n\\r\\nNew Logon:\\r\\n\\tSecurity ID:\\t\\ttest.com\\test106$\\r\\n\\tAccount Name:\\t\\ttest106$\\r\\n\\tAccount Domain:\\t\\ttest.com\\r\\n\\tLogon ID:\\t\\t0x9DE8990DE\\r\\n\\tLogon GUID:\\t\\t{E53069F0-662E-0C65-F889-AA8D8770D56A}\\r\\n\\r\\nProcess Information:\\r\\n\\tProcess ID:\\t\\t0x0\\r\\n\\tProcess Name:\\t\\t-\\r\\n\\r\\nNetwork Information:\\r\\n\\tWorkstation Name:\\t\\r\\n\\tSource Network Address:\\t100.00.100.1\\r\\n\\tSource Port:\\t\\t39028\\r\\n\\r\\nDetailed Authentication Information:\\r\\n\\tLogon Process:\\t\\tKerberos\\r\\n\\tAuthentication Package:\\tKerberos\\r\\n\\tTransited Services:\\t-\\r\\n\\tPackage Name (NTLM only):\\t-\\r\\n\\tKey Length:\\t\\t0\\r\\n\\r\\nThis event is generated when a logon session is created. It is generated on the computer that was accessed.\\r\\n\\r\\nThe subject fields indicate the account on the local system which requested the logon. This is most commonly a service such as the Server service, or a local process such as Winlogon.exe or Services.exe.\\r\\n\\r\\nThe logon type field indicates the kind of logon that occurred. The most common types are 2 (interactive) and 3 (network).\\r\\n\\r\\nThe New Logon fields indicate the account for whom the new logon was created, i.e. the account that was logged on.\\r\\n\\r\\nThe network fields indicate where a remote logon request originated. Workstation name is not always available and may be left blank in some cases.\\r\\n\\r\\nThe impersonation level field indicates the extent to which a process in the logon session can impersonate.\\r\\n\\r\\nThe authentication information fields provide detailed information about this specific logon request.\\r\\n\\t- Logon GUID is a unique identifier that can be used to correlate this event with a KDC event.\\r\\n\\t- Transited services indicate which intermediate services have participated in this logon request.\\r\\n\\t- Package name indicates which sub-protocol was used among the NTLM protocols.\\r\\n\\t- Key length indicates the length of the generated session key. This will be 0 if no session key was requested.","id":"c54d7f17-8eb8-4d78-a8f7-4b681256e2b3"}',
    '{"_raw":"04/03/2019 05:57:33 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4625\\nEventType=0\\nType=Information\\nComputerName=abc.test.com\\nTaskCategory=Logon\\nOpCode=Info\\nRecordNumber=849982687\\nKeywords=Audit Failure\\nMessage=An account failed to log on.\\r\\n\\r\\nSubject:\\r\\n\\tSecurity ID:\\t\\tNULL SID\\r\\n\\tAccount Name:\\t\\t-\\r\\n\\tAccount Domain:\\t\\t-\\r\\n\\tLogon ID:\\t\\t0x0\\r\\n\\r\\nLogon Type:\\t\\t\\t3\\r\\n\\r\\nAccount For Which Logon Failed:\\r\\n\\tSecurity ID:\\t\\tNULL SID\\r\\n\\tAccount Name:\\t\\thxyz\\r\\n\\tAccount Domain:\\t\\thxyz-PC1\\r\\n\\r\\nFailure Information:\\r\\n\\tFailure Reason:\\t\\tUnknown user name or bad password.\\r\\n\\tStatus:\\t\\t\\t0xc000006d\\r\\n\\tSub Status:\\t\\t0xc0000064\\r\\n\\r\\nProcess Information:\\r\\n\\tCaller Process ID:\\t0x0\\r\\n\\tCaller Process Name:\\t-\\r\\n\\r\\nNetwork Information:\\r\\n\\tWorkstation Name:\\thxyz-PC1\\r\\n\\tSource Network Address:\\t10.10.100.20\\r\\n\\tSource Port:\\t\\t53662\\r\\n\\r\\nDetailed Authentication Information:\\r\\n\\tLogon Process:\\t\\tNtLmSsp \\r\\n\\tAuthentication Package:\\tNTLM\\r\\n\\tTransited Services:\\t-\\r\\n\\tPackage Name (NTLM only):\\t-\\r\\n\\tKey Length:\\t\\t0\\r\\n\\r\\nThis event is generated when a logon request fails. It is generated on the computer where access was attempted.\\r\\n\\r\\nThe Subject fields indicate the account on the local system which requested the logon. This is most commonly a service such as the Server service, or a local process such as Winlogon.exe or Services.exe.\\r\\n\\r\\nThe Logon Type field indicates the kind of logon that was requested. The most common types are 2 (interactive) and 3 (network).\\r\\n\\r\\nThe Process Information fields indicate which account and process on the system requested the logon.\\r\\n\\r\\nThe Network Information fields indicate where a remote logon request originated. Workstation name is not always available and may be left blank in some cases.\\r\\n\\r\\nThe authentication information fields provide detailed information about this specific logon request.\\r\\n\\t- Transited services indicate which intermediate services have participated in this logon request.\\r\\n\\t- Package name indicates which sub-protocol was used among the NTLM protocols.\\r\\n\\t- Key length indicates the length of the generated session key. This will be 0 if no session key was requested.","sourcetype":"WinEventLog:Security","source":"WinEventLog:Security","_si":["sc.lab.test.com","wineventlog"],"_sourcetype":"WinEventLog:Security","Message":"An account failed to log on.\\r\\n\\r\\nSubject:\\r\\n\\tSecurity ID:\\t\\tNULL SID\\r\\n\\tAccount Name:\\t\\t-\\r\\n\\tAccount Domain:\\t\\t-\\r\\n\\tLogon ID:\\t\\t0x0\\r\\n\\r\\nLogon Type:\\t\\t\\t3\\r\\n\\r\\nAccount For Which Logon Failed:\\r\\n\\tSecurity ID:\\t\\tNULL SID\\r\\n\\tAccount Name:\\t\\thxyz\\r\\n\\tAccount Domain:\\t\\thxyz-PC1\\r\\n\\r\\nFailure Information:\\r\\n\\tFailure Reason:\\t\\tUnknown user name or bad password.\\r\\n\\tStatus:\\t\\t\\t0xc000006d\\r\\n\\tSub Status:\\t\\t0xc0000064\\r\\n\\r\\nProcess Information:\\r\\n\\tCaller Process ID:\\t0x0\\r\\n\\tCaller Process Name:\\t-\\r\\n\\r\\nNetwork Information:\\r\\n\\tWorkstation Name:\\thxyz-PC1\\r\\n\\tSource Network Address:\\t10.10.100.20\\r\\n\\tSource Port:\\t\\t53662\\r\\n\\r\\nDetailed Authentication Information:\\r\\n\\tLogon Process:\\t\\tNtLmSsp \\r\\n\\tAuthentication Package:\\tNTLM\\r\\n\\tTransited Services:\\t-\\r\\n\\tPackage Name (NTLM only):\\t-\\r\\n\\tKey Length:\\t\\t0\\r\\n\\r\\nThis event is generated when a logon request fails. It is generated on the computer where access was attempted.\\r\\n\\r\\nThe Subject fields indicate the account on the local system which requested the logon. This is most commonly a service such as the Server service, or a local process such as Winlogon.exe or Services.exe.\\r\\n\\r\\nThe Logon Type field indicates the kind of logon that was requested. The most common types are 2 (interactive) and 3 (network).\\r\\n\\r\\nThe Process Information fields indicate which account and process on the system requested the logon.\\r\\n\\r\\nThe Network Information fields indicate where a remote logon request originated. Workstation name is not always available and may be left blank in some cases.\\r\\n\\r\\nThe authentication information fields provide detailed information about this specific logon request.\\r\\n\\t- Transited services indicate which intermediate services have participated in this logon request.\\r\\n\\t- Package name indicates which sub-protocol was used among the NTLM protocols.\\r\\n\\t- Key length indicates the length of the generated session key. This will be 0 if no session key was requested.","_bkt":"wineventlog~313~8C261931-2C10-4450-B82C-39A63512E150","EventCode":"4625","_indextime":"1554242244","index":"wineventlog","_time":"1554242253","_pre_msg":"04/03/2019 05:57:33 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4625\\nEventType=0\\nType=Information\\nComputerName=abc.test.com\\nTaskCategory=Logon\\nOpCode=Info\\nRecordNumber=849982687\\nKeywords=Audit Failure","_cd":"313:1467779602","_serial":"16723","splunk_server":"sc.lab.test.com","host":"zjdhcp01","linecount":"61","id":"cf4876f3-716c-415c-994e-84acda054c9c"}',
    '{"_pre_msg":"04/03/2019 11:58:59 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=5156\\nEventType=0\\nType=Information\\nComputerName=user234.test.com\\nTaskCategory=Filtering Platform Connection\\nOpCode=Info\\nRecordNumber=241754521\\nKeywords=Audit Success","host":"user234","_raw":"04/03/2019 11:58:59 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=5156\\nEventType=0\\nType=Information\\nComputerName=user234.test.com\\nTaskCategory=Filtering Platform Connection\\nOpCode=Info\\nRecordNumber=241754521\\nKeywords=Audit Success\\nMessage=The Windows Filtering Platform has permitted a connection.\\r\\n\\r\\nApplication Information:\\r\\n\\tProcess ID:\\t\\t4\\r\\n\\tApplication Name:\\tSystem\\r\\n\\r\\nNetwork Information:\\r\\n\\tDirection:\\t\\tInbound\\r\\n\\tSource Address:\\t\\t100.20.100.20\\r\\n\\tSource Port:\\t\\t138\\r\\n\\tDestination Address:\\t100.20.100.30\\r\\n\\tDestination Port:\\t\\t138\\r\\n\\tProtocol:\\t\\t17\\r\\n\\r\\nFilter Information:\\r\\n\\tFilter Run-Time ID:\\t0\\r\\n\\tLayer Name:\\t\\tReceive/Accept\\r\\n\\tLayer Run-Time ID:\\t44","_si":["sc.lab.test.com","wineventlog"],"source":"WinEventLog:Security","sourcetype":"WinEventLog:Security","splunk_server":"sc.lab.test.com","_bkt":"wineventlog~316~8C261931-2C10-4450-B82C-39A63512E150","_sourcetype":"WinEventLog:Security","_indextime":"1554317930","EventCode":"5156","Message":"The Windows Filtering Platform has permitted a connection.\\r\\n\\r\\nApplication Information:\\r\\n\\tProcess ID:\\t\\t4\\r\\n\\tApplication Name:\\tSystem\\r\\n\\r\\nNetwork Information:\\r\\n\\tDirection:\\t\\tInbound\\r\\n\\tSource Address:\\t\\t100.20.100.20\\r\\n\\tSource Port:\\t\\t138\\r\\n\\tDestination Address:\\t100.20.100.30\\r\\n\\tDestination Port:\\t\\t138\\r\\n\\tProtocol:\\t\\t17\\r\\n\\r\\nFilter Information:\\r\\n\\tFilter Run-Time ID:\\t0\\r\\n\\tLayer Name:\\t\\tReceive/Accept\\r\\n\\tLayer Run-Time ID:\\t44","linecount":"29","_serial":"136","_cd":"316:913400766","index":"wineventlog","_time":"1554317939","id":"c3f48bba-90a1-4999-84a6-4da9d964d31d"}',
    '{"_pre_msg":"04/03/2019 11:58:59 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=5157\\nEventType=0\\nType=Information\\nComputerName=abc106.test.com\\nTaskCategory=Filtering Platform Connection\\nOpCode=Info\\nRecordNumber=2099763859\\nKeywords=Audit Failure","host":"abc106","_raw":"04/03/2019 11:58:59 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=5157\\nEventType=0\\nType=Information\\nComputerName=abc106.test.com\\nTaskCategory=Filtering Platform Connection\\nOpCode=Info\\nRecordNumber=2099763859\\nKeywords=Audit Failure\\nMessage=The Windows Filtering Platform has blocked a connection.\\r\\n\\r\\nApplication Information:\\r\\n\\tProcess ID:\\t\\t1048\\r\\n\\tApplication Name:\\t\\device\\harddiskvolume1\\windows\\system32\\svchost.exe\\r\\n\\r\\nNetwork Information:\\r\\n\\tDirection:\\t\\tInbound\\r\\n\\tSource Address:\\t\\t100.20.100.30\\r\\n\\tSource Port:\\t\\t137\\r\\n\\tDestination Address:\\t100.20.100.20\\r\\n\\tDestination Port:\\t\\t137\\r\\n\\tProtocol:\\t\\t0\\r\\n\\r\\nFilter Information:\\r\\n\\tFilter Run-Time ID:\\t65595\\r\\n\\tLayer Name:\\t\\tReceive/Accept\\r\\n\\tLayer Run-Time ID:\\t44","_si":["sc.lab.test.com","wineventlog"],"source":"WinEventLog:Security","sourcetype":"WinEventLog:Security","splunk_server":"sc.lab.test.com","_bkt":"wineventlog~316~8C261931-2C10-4450-B82C-39A63512E150","_sourcetype":"WinEventLog:Security","_indextime":"1554317931","EventCode":"5157","Message":"The Windows Filtering Platform has blocked a connection.\\r\\n\\r\\nApplication Information:\\r\\n\\tProcess ID:\\t\\t1048\\r\\n\\tApplication Name:\\t\\device\\harddiskvolume1\\windows\\system32\\svchost.exe\\r\\n\\r\\nNetwork Information:\\r\\n\\tDirection:\\t\\tInbound\\r\\n\\tSource Address:\\t\\t100.20.100.30\\r\\n\\tSource Port:\\t\\t137\\r\\n\\tDestination Address:\\t100.20.100.20\\r\\n\\tDestination Port:\\t\\t137\\r\\n\\tProtocol:\\t\\t0\\r\\n\\r\\nFilter Information:\\r\\n\\tFilter Run-Time ID:\\t65595\\r\\n\\tLayer Name:\\t\\tReceive/Accept\\r\\n\\tLayer Run-Time ID:\\t44","linecount":"29","_serial":"57","_cd":"316:913426654","index":"wineventlog","_time":"1554317939","id":"565beda9-346a-46a3-9f1f-25eab8d3414d"}',
    '{"_raw":"04/03/2019 05:57:33 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4798\\nEventType=0\\nType=Information\\nComputerName=abc.test.com\\nTaskCategory=Logon\\nOpCode=Info\\nRecordNumber=849982687\\nKeywords=Audit Failure\\nMessage=A user\'s local group membership was enumerated.\\r\\n\\r\\nSubject:\\r\\n\\tSecurity ID:\\t\\tNULL SID\\r\\n\\tAccount Name:\\t\\t-\\r\\n\\tAccount Domain:\\t\\t-\\r\\n\\tLogon ID:\\t\\t0x0\\r\\n\\r\\nUser:\\r\\n\\tSecurity ID:\\t\\tNULL SID\\r\\n\\tAccount Name:\\t\\thxyz\\r\\n\\tAccount Domain:\\t\\thxyz-PC1\\r\\n\\r\\nProcess Information:\\r\\n\\tProcess ID:\\t0x0\\r\\n\\tProcess Name:\\t-","sourcetype":"WinEventLog:Security","source":"WinEventLog:Security","_si":["sc.lab.test.com","wineventlog"],"_sourcetype":"WinEventLog:Security","Message":"A user\'s local group membership was enumerated.\\r\\n\\r\\nSubject:\\r\\n\\tSecurity ID:\\t\\tNULL SID\\r\\n\\tAccount Name:\\t\\t-\\r\\n\\tAccount Domain:\\t\\t-\\r\\n\\tLogon ID:\\t\\t0x0\\r\\n\\r\\nUser:\\r\\n\\tSecurity ID:\\t\\tNULL SID\\r\\n\\tAccount Name:\\t\\thxyz\\r\\n\\tAccount Domain:\\t\\thxyz-PC1\\r\\n\\r\\nProcess Information:\\r\\n\\tProcess ID:\\t0x0\\r\\n\\tProcess Name:\\t-","_bkt":"wineventlog~313~8C261931-2C10-4450-B82C-39A63512E150","EventCode":"4798","_indextime":"1554242244","index":"wineventlog","_time":"1554242253","_pre_msg":"04/03/2019 05:57:33 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4798\\nEventType=0\\nType=Information\\nComputerName=abc.test.com\\nTaskCategory=Logon\\nOpCode=Info\\nRecordNumber=849982687\\nKeywords=Audit Failure","_cd":"313:1467779602","_serial":"16723","splunk_server":"sc.lab.test.com","host":"zjdhcp01","linecount":"61","id":"cf4876f3-716c-415c-994e-84acda054c9c"}',
    '{"EventCode":"4769","_raw":"09/27/2018 04:45:36 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4769\\nEventType=0\\nType=Information\\nComputerName=test.localhost.com\\nTaskCategory=Kerberos Service Ticket Operations\\nOpCode=Info\\nRecordNumber=8876865135\\nKeywords=Audit Success\\nMessage=A Kerberos service ticket was requested.\\n\\nAccount Information:\\n\\tAccount Name:\\t\\tuser@localhost.com\\n\\tAccount Domain:\\t\\tlocalhost.com\\n\\tLogon GUID:\\t\\t{1F1D4C09-E154-4898-4EB8-E3A03E130D11}\\n\\nService Information:\\n\\tService Name:\\t\\ttest.localhost.com\\n\\tService ID:\\t\\tNONE_MAPPED\\n\\nNetwork Information:\\n\\tClient Address:\\t\\t::ffff:100.10.100.20\\n\\tClient Port:\\t\\t26061\\n\\nAdditional Information:\\n\\tTicket Options:\\t\\t0x40810000\\n\\tTicket Encryption Type:\\t0x17\\n\\tFailure Code:\\t\\t0x0\\n\\tTransited Services:\\t-\\n\\nThis event is generated every time access is requested to a resource such as a computer or a Windows service.  The service name indicates the resource to which access was requested.\\n\\nThis event can be correlated with Windows logon events by comparing the Logon GUID fields in each event.  The logon event occurs on the machine that was accessed, which is often a different machine than the domain controller which issued the service ticket.\\n\\nTicket options, encryption types, and failure codes are defined in RFC 4120.","id":"cf4876f3-716c-415c-994e-84acda054c9c"}',
    '{"EventCode":"4770","_raw":"09/27/2018 05:15:34 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4770\\nEventType=0\\nType=Information\\nComputerName=test.localhost.com\\nTaskCategory=Kerberos Service Ticket Operations\\nOpCode=Info\\nRecordNumber=186980567\\nKeywords=Audit Success\\nMessage=A Kerberos service ticket was renewed.\\n\\nAccount Information:\\n\\tAccount Name:\\t\\tTEST@LOCALHOST.COM\\n\\tAccount Domain:\\t\\tLOCALHOST.COM\\n\\nService Information:\\n\\tService Name:\\t\\tuser\\n\\tService ID:\\t\\tLOCALHOST.COM\\user\\n\\nNetwork Information:\\n\\tClient Address:\\t\\t::ffff:10.30.100.130\\n\\tClient Port:\\t\\t62133\\n\\nAdditional Information:\\n\\tTicket Options:\\t\\t0x50800002\\n\\tTicket Encryption Type:\\t0x12\\n\\nTicket options and encryption types are defined in RFC 4120.","id":"052b3a64-f1bd-4884-8e48-30b553bc495a"}',
    '{"id":"cf4876f3-716c-415c-994e-84acda054c9c","_sourcetype": "WinEventLog:Security", "linecount": "39", "index": "wineventlog", "_raw": "12/06/2018 06:52:05 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4771\\nEventType=0\\nType=Information\\nComputerName=test.localhost.com\\nTaskCategory=Kerberos Authentication Service\\nOpCode=Info\\nRecordNumber=4901782953\\nKeywords=Audit Failure\\nMessage=Kerberos pre-authentication failed.\\r\\n\\r\\nAccount Information:\\r\\n\\tSecurity ID:\\t\\tLOCALHOST.COM\\lab\\r\\n\\tAccount Name:\\t\\tlab\\r\\n\\r\\nService Information:\\r\\n\\tService Name:\\t\\tuser/LOCALHOST.COM\\r\\n\\r\\nNetwork Information:\\r\\n\\tClient Address:\\t\\t100.20.1.70\\r\\n\\tClient Port:\\t\\t60284\\r\\n\\r\\nAdditional Information:\\r\\n\\tTicket Options:\\t\\t0x40800000\\r\\n\\tFailure Code:\\t\\t0x18\\r\\n\\tPre-Authentication Type:\\t2\\r\\n\\r\\nCertificate Information:\\r\\n\\tCertificate Issuer Name:\\t\\t\\r\\n\\tCertificate Serial Number: \\t\\r\\n\\tCertificate Thumbprint:\\t\\t\\r\\n\\r\\nCertificate information is only provided if a certificate was used for pre-authentication.\\r\\n\\r\\nPre-authentication types, ticket options and failure codes are defined in RFC 4120.\\r\\n\\r\\nIf the ticket was malformed or damaged during transit and could not be decrypted, then many fields in this event might not be present.", "EventCode": "4771", "host": "BGDC101", "_indextime": "1544059330", "Message": "Kerberos pre-authentication failed.\\r\\n\\r\\nAccount Information:\\r\\n\\tSecurity ID:\\t\\tLOCALHOST.COM\\lab\\r\\n\\tAccount Name:\\t\\tlab\\r\\n\\r\\nService Information:\\r\\n\\tService Name:\\t\\tuser/LOCALHOST.COM\\r\\n\\r\\nNetwork Information:\\r\\n\\tClient Address:\\t\\t100.20.1.70\\r\\n\\tClient Port:\\t\\t60284\\r\\n\\r\\nAdditional Information:\\r\\n\\tTicket Options:\\t\\t0x40800000\\r\\n\\tFailure Code:\\t\\t0x18\\r\\n\\tPre-Authentication Type:\\t2\\r\\n\\r\\nCertificate Information:\\r\\n\\tCertificate Issuer Name:\\t\\t\\r\\n\\tCertificate Serial Number: \\t\\r\\n\\tCertificate Thumbprint:\\t\\t\\r\\n\\r\\nCertificate information is only provided if a certificate was used for pre-authentication.\\r\\n\\r\\nPre-authentication types, ticket options and failure codes are defined in RFC 4120.\\r\\n\\r\\nIf the ticket was malformed or damaged during transit and could not be decrypted, then many fields in this event might not be present.", "splunk_server": "localhost", "source": "WinEventLog:Security", "_cd": "215:335179321", "_serial": "0", "_bkt": "wineventlog~215~2CDBBBA3-F529-4047-AF8A-F1380407313B", "_time": "1544059325", "_si": ["localhost", "wineventlog"], "sourcetype": "WinEventLog:Security", "_pre_msg": "12/06/2018 06:52:05 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4771\\nEventType=0\\nType=Information\\nComputerName=test.localhost.com\\nTaskCategory=Kerberos Authentication Service\\nOpCode=Info\\nRecordNumber=4901782953\\nKeywords=Audit Failure"}',
    '{"EventCode":"4781","_raw":"09/27/2018 05:15:34 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4781\\nEventType=0\\nType=Information\\nComputerName=test.localhost.com\\nTaskCategory=Kerberos Service Ticket Operations\\nOpCode=Info\\nRecordNumber=186980567\\nKeywords=Audit Success\\nMessage=The name of an account was changed.\\n\\nSubject:\\n\\tSecurity ID:\\t\\tACME\Administrator\\n\\tAccount Name:\\t\\tTEST@LOCALHOST.COM\\n\\tAccount Domain:\\t\\tLOCALHOST.COM\\n\\tLogon ID:\\t\\t0x1f40f\\n\\nTarget Account:\\n\\tSecurity ID:\\t\\tACME\emp-nbonaparte\\n\\tAccount Domain:\\t\\tACME\\n\\tOld Account Name:\\t\\tnbonaparte\\n\\tNew Account Name:\\t\\temp-nbonaparte\\n\\nAdditional Information:\\n\\tPrivileges:\\t\\t-","id":"052b3a64-f1bd-4884-8e48-30b553bc495a"}',
    '{"EventCode":"4782","_raw":"09/27/2018 05:15:34 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4782\\nEventType=0\\nType=Information\\nComputerName=test.localhost.com\\nTaskCategory=Kerberos Service Ticket Operations\\nOpCode=Info\\nRecordNumber=186980567\\nKeywords=Audit Success\\n\\nSubject:\\n\\tSecurity ID:\\t\\tACME\Administrator\\n\\tAccount Name:\\t\\tTEST@LOCALHOST.COM\\n\\tAccount Domain:\\t\\tLOCALHOST.COM\\n\\tLogon ID:\\t\\t0x1f40f\\n\\nTarget Account:\\n\\tAccount Domain:\\t\\tACME\\n\\tAccount Name:\\t\\tnbonaparte","id":"052b3a64-f1bd-4884-8e48-30b553bc495a"}',
    '{"EventCode":"4634","_raw":"09/27/2018 05:15:34 AM\\\nLogName=Security\\\nSourceName=Microsoft Windows security auditing.\\\nEventCode=4634\\\nEventType=0\\\nType=Information\\\nComputerName=test.localhost.com\\\nTaskCategory=Kerberos Service Ticket Operations\\\nOpCode=Info\\\nRecordNumber=186980567\\\nKeywords=Audit Success\\\nMessage=An account was logged off.\\n\\nSubject:\\n\\tSecurity ID:\\t\\tANONYMOUS LOGON\\n\\tAccount Name:\\t\\tAppService\\n\\tAccount Domain:\\t\\tDomain001\\n\\tLogon ID:\\t\\t0x27b9013\\n\\nLogon Type:  3\\n\\nThis event is generated when a logon session is destroyed. It may be positively correlated with a logon event using the Logon ID value. Logon IDs are only unique between reboots on the same computer.","id":"052b3a64-f1bd-4884-8e48-30b553bc495a"}',
    '{"EventCode":"4647","_raw":"09/27/2018 05:15:34 AM\\\nLogName=Security\\\nSourceName=Microsoft Windows security auditing.\\\nEventCode=4647\\\nEventType=0\\\nType=Information\\\nComputerName=test.localhost.com\\\nTaskCategory=Kerberos Service Ticket Operations\\\nOpCode=Info\\\nRecordNumber=186980567\\\nKeywords=Audit Success\\\nMessage=User initiated logoff.\\n\\nSubject:\\n\\tSecurity ID:\\t\\tANONYMOUS LOGON\\n\\tAccount Name:\\t\\tAppService\\n\\tAccount Domain:\\t\\tDomain001\\n\\tLogon ID:\\t\\t0x27b9013\\n\\nThis event is generated when a logoff is initiated but the token reference count is not zero and the logon session cannot be destroyed.  No further user-initiated activity can occur.  This event can be interpreted as a logoff event.","id":"052b3a64-f1bd-4884-8e48-30b553bc495a"}',
    '{"EventCode":"4648","_raw":"09/27/2018 05:15:34 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4648\\nEventType=0\\nType=Information\\nComputerName=test.localhost.com\\nTaskCategory=Kerberos Service Ticket Operations\\nOpCode=Info\\nRecordNumber=186980567\\nKeywords=Audit Success\\nMessage==A logon was attempted using explicit credentials.\\n\\nSubject:\\n\\tSecurity ID:\\t\\tWIN-R9H529RIO4Y\Administrator\\n\\tAccount Name:\\t\\tAdministrator\\n\\tAccount Domain:\\t\\tWIN-R9H529RIO4Y\\n\\tLogon ID:\\t\\t0x1ba0e\\n\\tLogon GUID:\\t\\t  {00000000-0000-0000-0000-000000000000}\\n\\nAccount Whose Credentials Were Used:\\n\\tAccount Name:\\t\\trsmith@mtg.com\\n\\tAccount Domain:\\t\\tWIN-R9H529RIO4Y\\n\\tLogon GUID:\\t\\t{00000000-0000-0000-0000-000000000000}\\n\\nTarget Server:\\n\\tTarget Server Name:\\t\\tsp01.IceMAIL.com\\n\\tAdditional Information:\\t\\tsp01.IceMAIL.com\\n\\nProcess Information:\\n\\tProcess ID:\\t\\t0x77c\\n\\tProcess Name:\\t\\tC:\\t\\t\Program Files\Internet Explorer\iexplore.exe\\n\\nNetwork Information:\\n\\tNetwork Address:-\\n\\tPort:-","id":\\t\\t"052b3a64-f1bd-4884-8e48-30b553bc495a"}',
    '{"EventCode":"4672","_raw":"09/27/2018 10:52:50 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4672\\nEventType=0\\nType=Information\\nComputerName=test.localhost.com\\nTaskCategory=Special Logon\\nOpCode=Info\\nRecordNumber=3706115579\\nKeywords=Audit Success\\nMessage=Special privileges assigned to new logon.\\n\\nSubject:\\n\\tSecurity ID:\\t\\tDEV\\tuser\\n\\tAccount Name:\\t\\tuser\\n\\tAccount Domain:\\t\\tDEV\\n\\tLogon ID:\\t\\t0x800A513D\\n\\nPrivileges:\\t\\tSeSecurityPrivilege\\n\\t\\t\\tSeBackupPrivilege\\n\\t\\t\\tSeRestorePrivilege\\n\\t\\t\\tSeTakeOwnershipPrivilege\\n\\t\\t\\tSeDebugPrivilege\\n\\t\\t\\tSeSystemEnvironmentPrivilege\\n\\t\\t\\tSeLoadDriverPrivilege\\n\\t\\t\\tSeImpersonatePrivilege","id":"052b3a64-f1bd-4884-8e48-30b553bc495a"}',
    '{"Account_Domain": "test.com", "Account_Name": "fvjbvfjbvf$", "ComputerName": "fvjbvfjbvf.test.com", "Logon_ID": "0x3e7", "Message": "A privileged service was called.\\n\\nSubject:\\n\\tSecurity ID:\\t\\tNT AUTHORITY\\SYSTEM\\n\\tAccount Name:\\t\\tfvjbvfjbvf$\\n\\tAccount Domain:\\t\\ttest.com\\n\\tLogon ID:\\t\\t0x3e7\\n\\nService:\\n\\tServer:\\tNT Local Security Authority / Authentication Service\\n\\tService Name:\\tLsaRegisterLogonProcess()\\n\\nProcess:\\n\\tProcess ID:\\t0x234\\n\\tProcess Name:\\tC:\\Windows\\System32\\lsass.exe\\n\\nService Request Information:\\n\\tPrivileges:\\t\\tSeTcbPrivilege", "Security_ID": "NT AUTHORITY\\SYSTEM", "_bkt": "wineventlog~15~3D7EB920-B824-4467-A0DA-EFE0925C0D7D", "_cd": "15:36073965", "_indextime": "1527787976", "_pre_msg": "04/30/2018 05:13:59 PM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4673\\nEventType=0\\nType=Information\\nComputerName=fvjbvfjbvf.test.com\\nTaskCategory=Sensitive Privilege Use\\nOpCode=Info\\nRecordNumber=6623591495\\nKeywords=Audit Success", "_raw":"04/30/2018 05:13:59 PM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4673\\nEventType=0\\nType=Information\\nComputerName=fvjbvfjbvf.test.com\\nTaskCategory=Sensitive Privilege Use\\nOpCode=Info\\nRecordNumber=6623591495\\nKeywords=Audit Success\\nMessage=A privileged service was called.\\n\\nSubject:\\n\\tSecurity ID:\\t\\tNT AUTHORITY\\SYSTEM\\n\\tAccount Name:\\t\\tfvjbvfjbvf$\\n\\tAccount Domain:\\t\\ttest.com\\n\\tLogon ID:\\t\\t0x3e7\\n\\nService:\\n\\tServer:\\tNT Local Security Authority / Authentication Service\\n\\tService Name:\\tLsaRegisterLogonProcess()\\n\\nProcess:\\n\\tProcess ID:\\t0x234\\n\\tProcess Name:\\tC:\\Windows\\System32\\lsass.exe\\n\\nService Request Information:\\n\\tPrivileges:\\t\\tSeTcbPrivilege", "_serial": "153", "_si": ["idx9.nvda-sec.splunkcloud.com", "wineventlog"], "_sourcetype": "WinEventLog:Security", "_time": "2018-05-01T00:13:59.000+00:00", "dest_nt_host": "fvjbvfjbvf.test.com", "host": "hqdvppmwb07", "index": "wineventlog", "linecount": "29", "source": "WinEventLog:Security", "sourcetype": "WinEventLog:Security", "splunk_server": "sc.lab.test.com, "vendor_privilege": "SeTcbPrivilege","id":"sdgfhsdfhj-3245-dsf"}',
    '{"preview":false,"result":{"EventCode":"4722","_raw":"09/27/2018 09:56:10 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4722\\nEventType=0\\nType=Information\\nComputerName=localhost.test.com\\nTaskCategory=User Account Management\\nOpCode=Info\\nRecordNumber=37352998061\\nKeywords=Audit Success\\nMessage=A user account was enabled.\\n\\nSubject:\\n\\tSecurity ID:\\t\\ttest.com\\dhgfckkcg\\n\\tAccount Name:\\t\\tdhgfckkcg\\n\\tAccount Domain:\\t\\ttest.com\\n\\tLogon ID:\\t\\t0x2D55E5EF7\\n\\nTarget Account:\\n\\tSecurity ID:\\t\\ttest.com\\hgcghjj\\n\\tAccount Name:\\t\\thgcghjj\\n\\tAccount Domain:\\t\\ttest.com","id":"tdr5d-fjfgg-687bv-klhk"}',
    '{"preview":false,"result":{"EventCode":"4720","_raw":"09/27/2018 09:56:10 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4720\\nEventType=0\\nType=Information\\nComputerName=localhost.test.com\\nTaskCategory=User Account Management\\nOpCode=Info\\nRecordNumber=37352998061\\nKeywords=Audit Success\\nMessage=A user account was created.\\n\\nSubject:\\n\\tSecurity ID:\\t\\tACME-FR\administrator\\n\\tAccount Name:\\t\\tadministrator\\n\\tAccount Domain:\\t\\tACME-FR\\n\\tLogon ID:\\t\\t0x20f9d\\n\\nNew Account:\\n\\tSecurity ID:\\t\\tACME-FR\John.LockeAccount\\n\\tName:\\t\\tJohn.Locke\\n\\tAccount Domain:\\t\\tACME-FR\\n\\nAttributes:\\n\\tSAM Account Name:\\t\\tJohn.Locke\\n\\tDisplay Name:\\t\\tJohn Locke\\n\\tUser Principal Name:\\t\\tJohn.Locke@acme-fr.local\\n\\tHome Directory:\\t\\t-\\n\\tHome Drive:\\t\\t-\\n\\tScript Path:\\t\\t-\\n\\tProfile Path:\\t\\t-\\n\\tUser Workstations:\\t\\t-\\n\\tPassword Last Set:\\t\\t<never>\\n\\tAccount Expires:\\t\\t<never>\\n\\tPrimary Group ID:\\t\\t513\\n\\tAllowed To Delegate To:\\t\\t-\\n\\tOld UAC Value:\\t\\t0x0\\n\\tNew UAC Value:\\t\\t0x15\\n\\tUser Account Control:\\t\\t\\nAccount Disabled\\n\'Password Not Required\' - Enabled\\n\'Normal Account\' - Enabled\\n\\tUser Parameters:\\t\\t-\\n\\tSID History:\\t\\t-\\n\\tLogon Hours:\\t\\t<value not set>\\n\\nAdditional Information:\\n\\tPrivileges\\t\\t-","id":"tdr5d-fjfgg-687bv-klhk"}',
    '{"preview":false,"result":{"EventCode":"4723","_raw":"09/27/2018 10:24:34 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4723\\nEventType=0\\nType=Information\\nComputerName=test.localhost.com\\nTaskCategory=User Account Management\\nOpCode=Info\\nRecordNumber=9342213186\\nKeywords=Audit Failure\\nMessage=An attempt was made to change an account\'s password.\\n\\nSubject:\\n\\tSecurity ID:\\t\\ttest.com\\auser\\n\\tAccount Name:\\t\\tauser\\n\\tAccount Domain:\\t\\ttest.com\\n\\tLogon ID:\\t\\t0x258440926\\n\\nTarget Account:\\n\\tSecurity ID:\\t\\ttest.com\\auser\\n\\tAccount Name:\\t\\tauser\\n\\tAccount Domain:\\t\\ttest.com\\n\\nAdditional Information:\\n\\tPrivileges\\t\\t-","id":"tdr5d-fjfgg-687bv-klhk"}',
    '{"preview":false,"result":{"EventCode":"4724","_raw":"09/27/2018 10:24:34 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4724\\nEventType=0\\nType=Information\\nComputerName=test.localhost.com\\nTaskCategory=User Account Management\\nOpCode=Info\\nRecordNumber=9342213186\\nKeywords=Audit Failure\\nMessage=An attempt was made to reset an account\'s password.\\n\\nSubject:\\n\\tSecurity ID:\\t\\ttest.com\\auser\\n\\tAccount Name:\\t\\tauser\\n\\tAccount Domain:\\t\\ttest.com\\n\\tLogon ID:\\t\\t0x258440926\\n\\nTarget Account:\\n\\tSecurity ID:\\t\\ttest.com\\auser\\n\\tAccount Name:\\t\\tauser\\n\\tAccount Domain:\\t\\ttest.com","id":"tdr5d-fjfgg-687bv-klhk"}',
    '{"preview":false,"result":{"EventCode":"4725","_raw":"09/27/2018 10:24:34 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4725\\nEventType=0\\nType=Information\\nComputerName=test.localhost.com\\nTaskCategory=User Account Management\\nOpCode=Info\\nRecordNumber=9342213186\\nKeywords=Audit Failure\\nMessage=A user account was disabled..\\n\\nSubject:\\n\\tSecurity ID:\\t\\ttest.com\\auser\\n\\tAccount Name:\\t\\tauser\\n\\tAccount Domain:\\t\\ttest.com\\n\\tLogon ID:\\t\\t0x258440926\\n\\nTarget Account:\\n\\tSecurity ID:\\t\\ttest.com\\auser\\n\\tAccount Name:\\t\\tauser\\n\\tAccount Domain:\\t\\ttest.com","id":"tdr5d-fjfgg-687bv-klhk"}',
    '{"preview":false,"result":{"EventCode":"4726","_raw":"09/27/2018 10:24:34 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4726\\nEventType=0\\nType=Information\\nComputerName=test.localhost.com\\nTaskCategory=User Account Management\\nOpCode=Info\\nRecordNumber=9342213186\\nKeywords=Audit Failure\\nMessage=A user account was deleted.\\n\\nSubject:\\n\\tSecurity ID:\\t\\ttest.com\\auser\\n\\tAccount Name:\\t\\tauser\\n\\tAccount Domain:\\t\\ttest.com\\n\\tLogon ID:\\t\\t0x258440926\\n\\nTarget Account:\\n\\tSecurity ID:\\t\\ttest.com\\auser\\n\\tAccount Name:\\t\\tauser\\n\\tAccount Domain:\\t\\ttest.com\\n\\nAdditional Information:\\n\\tPrivileges\\t\\t-","id":"tdr5d-fjfgg-687bv-klhk"}',
    '{"EventCode":"4732","_raw":"09/19/2018 06:18:24 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4732\\nEventType=0\\nType=Information\\nComputerName=testuser.localhost.com\\nTaskCategory=Security Group Management\\nOpCode=Info\\nRecordNumber=7984447290\\nKeywords=Audit Success\\nMessage=A member was added to a security-enabled local group.\\n\\nSubject:\\n\\tSecurity ID:\\t\\tNT AUTHORITY\\SYSTEM\\n\\tAccount Name:\\t\\ttestuser$\\n\\tAccount Domain:\\t\\tlocalhost.COM\\n\\tLogon ID:\\t\\t0x3e7\\n\\nMember:\\n\\tSecurity ID:\\t\\tlocalhost.COM\\NV-LocalAdmins\\n\\tAccount Name:\\t\\t-\\n\\nGroup:\\n\\tSecurity ID:\\t\\ttestuser\\Offer Remote Assistance Helpers\\n\\tGroup Name:\\t\\tOffer Remote Assistance Helpers\\n\\tGroup Domain:\\t\\ttestuser\\n\\nAdditional Information:\\n\\tPrivileges:\\t\\t-","id":"tdr5d-fjfgg-687bv-klhk"}',
    '{"Account_Domain": ["PROD", "esfdhf06"], "Account_Name": ["esfdhf06$", "MATBLE"], "ComputerName": "esfdhf06.prod.test.com", "Logon_ID": "0x3E7", "Message": "A user account was changed.\\n\\nSubject:\\n\\tSecurity ID:\\t\\tNT AUTHORITY\\SYSTEM\\n\\tAccount Name:\\t\\tesfdhf06$\\n\\tAccount Domain:\\t\\tPROD\\n\\tLogon ID:\\t\\t0x3E7\\n\\nTarget Account:\\n\\tSecurity ID:\\t\\tesfdhf06\\ATBLE\\n\\tAccount Name:\\t\\tATBLE\\n\\tAccount Domain:\\t\\tesfdhf06\\n\\nChanged Attributes:\\n\\tSAM Account Name:\\tATBLE\\n\\tDisplay Name:\\t\\tMIKE ATBLE\\n\\tUser Principal Name:\\t-\\n\\tHome Directory:\\t\\t<value not set>\\n\\tHome Drive:\\t\\t<value not set>\\n\\tScript Path:\\t\\t<value not set>\\n\\tProfile Path:\\t\\t<value not set>\\n\\tUser Workstations:\\t<value not set>\\n\\tPassword Last Set:\\t5/1/2018 5:41:37 AM\\n\\tAccount Expires:\\t\\t<never>\\n\\tPrimary Group ID:\\t513\\n\\tAllowedToDelegateTo:\\t-\\n\\tOld UAC Value:\\t\\t0x210\\n\\tNew UAC Value:\\t\\t0x210\\n\\tUser Account Control:\\t-\\n\\tUser Parameters:\\t-\\n\\tSID History:\\t\\t-\\n\\tLogon Hours:\\t\\tAll\\n\\nAdditional Information:\\n\\tPrivileges:\\t\\t-", "Security_ID": ["NT AUTHORITY\\SYSTEM", "esfdhf06\\ATBLE"], "_bkt": "wineventlog~0~3D7EB920-B824-4467-A0DA-EFE0925C0D7D", "_cd": "0:1057390650", "_indextime": "1526126427", "_pre_msg": "05/01/2018 05:41:37 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4738\\nEventType=0\\nType=Information\\nComputerName=esfdhf06.prod.test.com\\nTaskCategory=User Account Management\\nOpCode=Info\\nRecordNumber=92255448\\nKeywords=Audit Success", "_raw":"05/01/2018 05:41:37 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4738\\nEventType=0\\nType=Information\\nComputerName=esfdhf06.prod.test.com\\nTaskCategory=User Account Management\\nOpCode=Info\\nRecordNumber=92255448\\nKeywords=Audit Success\\nMessage=A user account was changed.\\n\\nSubject:\\n\\tSecurity ID:\\t\\tNT AUTHORITY\\SYSTEM\\n\\tAccount Name:\\t\\tesfdhf06$\\n\\tAccount Domain:\\t\\tPROD\\n\\tLogon ID:\\t\\t0x3E7\\n\\nTarget Account:\\n\\tSecurity ID:\\t\\tesfdhf06\\ATBLE\\n\\tAccount Name:\\t\\tATBLE\\n\\tAccount Domain:\\t\\tesfdhf06\\n\\nChanged Attributes:\\n\\tSAM Account Name:\\tATBLE\\n\\tDisplay Name:\\t\\tMIKE ATBLE\\n\\tUser Principal Name:\\t-\\n\\tHome Directory:\\t\\t<value not set>\\n\\tHome Drive:\\t\\t<value not set>\\n\\tScript Path:\\t\\t<value not set>\\n\\tProfile Path:\\t\\t<value not set>\\n\\tUser Workstations:\\t<value not set>\\n\\tPassword Last Set:\\t5/1/2018 5:41:37 AM\\n\\tAccount Expires:\\t\\t<never>\\n\\tPrimary Group ID:\\t513\\n\\tAllowedToDelegateTo:\\t-\\n\\tOld UAC Value:\\t\\t0x210\\n\\tNew UAC Value:\\t\\t0x210\\n\\tUser Account Control:\\t-\\n\\tUser Parameters:\\t-\\n\\tSID History:\\t\\t-\\n\\tLogon Hours:\\t\\tAll\\n\\nAdditional Information:\\n\\tPrivileges:\\t\\t-", "_serial": "551", "_si": ["test.splunkcloud.com", "wineventlog"], "_sourcetype": "WinEventLog:Security", "_time": "2018-05-01T00:11:37.000+00:00", "dest_nt_host": "esfdhf06.prod.test.com", "host": "esfdhf06", "index": "wineventlog", "linecount": "46", "source": "WinEventLog:Security", "sourcetype": "WinEventLog:Security", "splunk_server": "test.splunkcloud.com", "vendor_privilege": "-","id":"tdr5d-fjfgg-687bv-klhk"}',
    '{"preview":false,"result":{"EventCode":"4740","_raw":"09/28/2018 01:53:37 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4740\\nEventType=0\\nType=Information\\nComputerName=sdgbjsd02.test.localhost.com\\nTaskCategory=User Account Management\\nOpCode=Info\\nRecordNumber=20832836\\nKeywords=Audit Success\\nMessage=A user account was locked out.\\n\\nSubject:\\n\\tSecurity ID:\\t\\tNT AUTHORITY\\SYSTEM\\n\\tAccount Name:\\t\\tsdgbjsd02$\\n\\tAccount Domain:\\t\\tNVDMZ\\n\\tLogon ID:\\t\\t0x3E7\\n\\nAccount That Was Locked Out:\\n\\tSecurity ID:\\t\\tsdgbjsd02\\Guest\\n\\tAccount Name:\\t\\tGuest\\n\\nAdditional Information:\\n\\tCaller Computer Name:\\tsdgbjsd01","id":"tdr5d-fjfgg-687bv-klhk"}',
    '{"preview":false,"result":{"EventCode":"4743","_raw":"09/27/2018 10:24:34 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4743\\nEventType=0\\nType=Information\\nComputerName=test.localhost.com\\nTaskCategory=User Account Management\\nOpCode=Info\\nRecordNumber=9342213186\\nKeywords=Audit Failure\\nMessage=A computer account was deleted.\\n\\nSubject:\\n\\tSecurity ID:\\t\\ttest.com\\auser\\n\\tAccount Name:\\t\\tauser\\n\\tAccount Domain:\\t\\ttest.com\\n\\tLogon ID:\\t\\t0x258440926\\n\\nTarget Account:\\n\\tSecurity ID:\\t\\ttest.com\\auser\\n\\tAccount Name:\\t\\tauser\\n\\tAccount Domain:\\t\\ttest.com\\n\\nAdditional Information:\\n\\tPrivileges\\t\\t-","id":"tdr5d-fjfgg-687bv-klhk"}',
    '{"EventCode":"4756","_raw":"09/19/2018 06:18:24 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4756\\nEventType=0\\nType=Information\\nComputerName=testuser.localhost.com\\nTaskCategory=Security Group Management\\nOpCode=Info\\nRecordNumber=7984447290\\nKeywords=Audit Success\\nMessage=A member was added to a security-enabled universal group.\\n\\nSubject:\\n\\tSecurity ID:\\t\\tNT AUTHORITY\\SYSTEM\\n\\tAccount Name:\\t\\ttestuser$\\n\\tAccount Domain:\\t\\tlocalhost.COM\\n\\tLogon ID:\\t\\t0x3e7\\n\\nMember:\\n\\tSecurity ID:\\t\\tlocalhost.COM\\NV-LocalAdmins\\n\\tAccount Name:\\t\\t-\\n\\nGroup:\\n\\tSecurity ID:\\t\\ttestuser\\Offer Remote Assistance Helpers\\n\\tGroup Name:\\t\\tOffer Remote Assistance Helpers\\n\\tGroup Domain:\\t\\ttestuser\\n\\nAdditional Information:\\n\\tPrivileges:\\t\\t-\\n\\tExpiration time:\\t\\t","id":"tdr5d-fjfgg-687bv-klhk"}',
    '{"preview":false,"result":{"EventCode":"4725","_raw":"09/27/2018 10:24:34 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4725\\nEventType=0\\nType=Information\\nComputerName=test.localhost.com\\nTaskCategory=User Account Management\\nOpCode=Info\\nRecordNumber=9342213186\\nKeywords=Audit Failure\\nMessage=A user account was unlocked.\\n\\nSubject:\\n\\tSecurity ID:\\t\\ttest.com\\auser\\n\\tAccount Name:\\t\\tauser\\n\\tAccount Domain:\\t\\ttest.com\\n\\tLogon ID:\\t\\t0x258440926\\n\\nTarget Account:\\n\\tSecurity ID:\\t\\ttest.com\\auser\\n\\tAccount Name:\\t\\tauser\\n\\tAccount Domain:\\t\\ttest.com","id":"tdr5d-fjfgg-687bv-klhk"}',
    '{"preview":false,"result":{"EventCode":"4768","_raw":"09/27/2018 09:08:02 AM\\nLogName=Security\\nSourceName=Microsoft Windows security auditing.\\nEventCode=4768\\nEventType=0\\nType=Information\\nComputerName=test02.localhost.com\\nTaskCategory=Kerberos Authentication Service\\nOpCode=Info\\nRecordNumber=1376039507\\nKeywords=Audit Success\\nMessage=A Kerberos authentication ticket (TGT) was requested.\\n\\nAccount Information:\\n\\tAccount Name:\\t\\tHealthMailbox06ca30c\\n\\tSupplied Realm Name:\\tlocalhost.com\\n\\tUser ID:\\t\\t\\tlocalhost.com\\HealthMailbox06ca30c\\n\\nService Information:\\n\\tService Name:\\t\\tasdfgrvk\\n\\tService ID:\\t\\tlocalhost.com\\asdfgrvk\\n\\nNetwork Information:\\n\\tClient Address:\\t\\t::ffff:10.20.90.30\\n\\tClient Port:\\t\\t6349\\n\\nAdditional Information:\\n\\tTicket Options:\\t\\t0x40810010\\n\\tResult Code:\\t\\t0x0\\n\\tTicket Encryption Type:\\t0x12\\n\\tPre-Authentication Type:\\t2\\n\\nCertificate Information:\\n\\tCertificate Issuer Name:\\t\\t\\n\\tCertificate Serial Number:\\t\\n\\tCertificate Thumbprint:\\t\\t\\n\\nCertificate information is only provided if a certificate was used for pre-authentication.\\n\\nPre-authentication types, ticket options, encryption types and result codes are defined in RFC 4120.","id":"asd-eter-34235-fgd-346"}',
]


def validate_4624(parsed_rec):
    assert parsed_rec["time"] == "04/01/2019 07:07:21 pm"
    assert parsed_rec["id"] == "c54d7f17-8eb8-4d78-a8f7-4b681256e2b3"
    assert parsed_rec["eventcode"] == "4624"
    assert (parsed_rec["detailed_authentication_information_authentication_package"] == "kerberos")
    assert (parsed_rec["new_logon_logon_guid"] == "{e53069f0-662e-0c65-f889-aa8d8770d56a}")
    assert parsed_rec["failure_information_failure_reason"] == ""
    assert parsed_rec["failure_information_status"] == ""
    assert parsed_rec["computername"] == ""
    assert parsed_rec["new_logon_logon_id"] == "0x9de8990de"
    assert parsed_rec["subject_security_id"] == "null sid"
    assert (parsed_rec["detailed_authentication_information_package_name_ntlm_only"] == "-")
    assert parsed_rec["logon_type"] == "3"
    assert parsed_rec["account_for_which_logon_failed_security_id"] == ""
    assert parsed_rec["detailed_authentication_information_key_length"] == "0"
    assert parsed_rec["subject_logon_id"] == "0x0"
    assert parsed_rec["process_information_caller_process_name"] == ""
    assert parsed_rec["process_information_caller_process_id"] == ""
    assert parsed_rec["subject_account_name"] == "-"
    assert parsed_rec["process_information_process_name"] == "-"
    assert parsed_rec["new_logon_account_name"] == "test106$"
    assert parsed_rec["process_information_process_id"] == "0x0"
    assert parsed_rec["failure_information_sub_status"] == ""
    assert parsed_rec["new_logon_security_id"] == "test.comest"
    assert parsed_rec["network_information_source_network_address"] == "100.00.100.1"
    assert parsed_rec["detailed_authentication_information_transited_services"] == "-"
    assert parsed_rec["new_logon_account_domain"] == "test.com"
    assert parsed_rec["subject_account_domain"] == "-"
    assert parsed_rec["detailed_authentication_information_logon_process"] == "kerberos"
    assert parsed_rec["account_for_which_logon_failed_account_domain"] == ""
    assert parsed_rec["account_for_which_logon_failed_account_name"] == ""
    assert parsed_rec["network_information_workstation_name"] == ""
    assert parsed_rec["network_information_source_port"] == "39028"
    assert parsed_rec["application_information_process_id"] == ""
    assert parsed_rec["application_information_application_name"] == ""
    assert parsed_rec["network_information_direction"] == ""
    assert parsed_rec["network_information_source_address"] == ""
    assert parsed_rec["network_information_destination_address"] == ""
    assert parsed_rec["network_information_destination_port"] == ""
    assert parsed_rec["network_information_protocol"] == ""
    assert parsed_rec["filter_information_filter_run_time_id"] == ""
    assert parsed_rec["filter_information_layer_name"] == ""
    assert parsed_rec["filter_information_layer_run_time_id"] == ""


def validate_4625(parsed_rec):
    assert parsed_rec["time"] == "04/03/2019 05:57:33 am"
    assert parsed_rec["id"] == "cf4876f3-716c-415c-994e-84acda054c9c"
    assert parsed_rec["eventcode"] == "4625"
    assert (parsed_rec["detailed_authentication_information_authentication_package"] == "ntlm")
    assert parsed_rec["new_logon_logon_guid"] == ""
    assert (parsed_rec["failure_information_failure_reason"] == "unknown user name or bad password.")
    assert parsed_rec["failure_information_status"] == "0xc000006d"
    assert parsed_rec["computername"] == "abc.test.com"
    assert parsed_rec["new_logon_logon_id"] == ""
    assert parsed_rec["subject_security_id"] == "null sid"
    assert (parsed_rec["detailed_authentication_information_package_name_ntlm_only"] == "-")
    assert parsed_rec["logon_type"] == "3"
    assert parsed_rec["account_for_which_logon_failed_security_id"] == "null sid"
    assert parsed_rec["detailed_authentication_information_key_length"] == "0"
    assert parsed_rec["subject_logon_id"] == "0x0"
    assert parsed_rec["process_information_caller_process_name"] == "-"
    assert parsed_rec["process_information_caller_process_id"] == "0x0"
    assert parsed_rec["subject_account_name"] == "-"
    assert parsed_rec["process_information_process_name"] == ""
    assert parsed_rec["new_logon_account_name"] == ""
    assert parsed_rec["process_information_process_id"] == ""
    assert parsed_rec["failure_information_sub_status"] == "0xc0000064"
    assert parsed_rec["new_logon_security_id"] == ""
    assert parsed_rec["network_information_source_network_address"] == "10.10.100.20"
    assert parsed_rec["detailed_authentication_information_transited_services"] == "-"
    assert parsed_rec["new_logon_account_domain"] == ""
    assert parsed_rec["subject_account_domain"] == "-"
    assert parsed_rec["detailed_authentication_information_logon_process"] == "ntlmssp"
    assert parsed_rec["account_for_which_logon_failed_account_domain"] == "hxyz"
    assert parsed_rec["account_for_which_logon_failed_account_name"] == "hxyz"
    assert parsed_rec["network_information_workstation_name"] == "hxyz-pc1"
    assert parsed_rec["network_information_source_port"] == "53662"
    assert parsed_rec["application_information_process_id"] == ""
    assert parsed_rec["application_information_application_name"] == ""
    assert parsed_rec["network_information_direction"] == ""
    assert parsed_rec["network_information_source_address"] == ""
    assert parsed_rec["network_information_destination_address"] == ""
    assert parsed_rec["network_information_destination_port"] == ""
    assert parsed_rec["network_information_protocol"] == ""
    assert parsed_rec["filter_information_filter_run_time_id"] == ""
    assert parsed_rec["filter_information_layer_name"] == ""
    assert parsed_rec["filter_information_layer_run_time_id"] == ""


def validate_5156(parsed_rec):
    assert parsed_rec["time"] == "04/03/2019 11:58:59 am"
    assert parsed_rec["id"] == "c3f48bba-90a1-4999-84a6-4da9d964d31d"
    assert parsed_rec["eventcode"] == "5156"
    assert (parsed_rec["detailed_authentication_information_authentication_package"] == "")
    assert parsed_rec["new_logon_logon_guid"] == ""
    assert parsed_rec["failure_information_failure_reason"] == ""
    assert parsed_rec["failure_information_status"] == ""
    assert parsed_rec["computername"] == ""
    assert parsed_rec["new_logon_logon_id"] == ""
    assert parsed_rec["subject_security_id"] == ""
    assert (parsed_rec["detailed_authentication_information_package_name_ntlm_only"] == "")
    assert parsed_rec["logon_type"] == ""
    assert parsed_rec["account_for_which_logon_failed_security_id"] == ""
    assert parsed_rec["detailed_authentication_information_key_length"] == ""
    assert parsed_rec["subject_logon_id"] == ""
    assert parsed_rec["process_information_caller_process_name"] == ""
    assert parsed_rec["process_information_caller_process_id"] == ""
    assert parsed_rec["subject_account_name"] == ""
    assert parsed_rec["process_information_process_name"] == ""
    assert parsed_rec["new_logon_account_name"] == ""
    assert parsed_rec["process_information_process_id"] == ""
    assert parsed_rec["failure_information_sub_status"] == ""
    assert parsed_rec["new_logon_security_id"] == ""
    assert parsed_rec["network_information_source_network_address"] == ""
    assert parsed_rec["detailed_authentication_information_transited_services"] == ""
    assert parsed_rec["new_logon_account_domain"] == ""
    assert parsed_rec["subject_account_domain"] == ""
    assert parsed_rec["detailed_authentication_information_logon_process"] == ""
    assert parsed_rec["account_for_which_logon_failed_account_domain"] == ""
    assert parsed_rec["account_for_which_logon_failed_account_name"] == ""
    assert parsed_rec["network_information_workstation_name"] == ""
    assert parsed_rec["network_information_source_port"] == "138"
    assert parsed_rec["application_information_process_id"] == "4"
    assert parsed_rec["application_information_application_name"] == "system"
    assert parsed_rec["network_information_direction"] == "inbound"
    assert parsed_rec["network_information_source_address"] == "100.20.100.20"
    assert parsed_rec["network_information_destination_address"] == "100.20.100.30"
    assert parsed_rec["network_information_destination_port"] == "138"
    assert parsed_rec["network_information_protocol"] == "17"
    assert parsed_rec["filter_information_filter_run_time_id"] == "0"
    assert parsed_rec["filter_information_layer_name"] == "receive/accept"
    assert parsed_rec["filter_information_layer_run_time_id"] == "44"


def validate_5157(parsed_rec):
    assert parsed_rec["time"] == "04/03/2019 11:58:59 am"
    assert parsed_rec["id"] == "565beda9-346a-46a3-9f1f-25eab8d3414d"
    assert parsed_rec["eventcode"] == "5157"
    assert (parsed_rec["detailed_authentication_information_authentication_package"] == "")
    assert parsed_rec["new_logon_logon_guid"] == ""
    assert parsed_rec["failure_information_failure_reason"] == ""
    assert parsed_rec["failure_information_status"] == ""
    assert parsed_rec["computername"] == ""
    assert parsed_rec["new_logon_logon_id"] == ""
    assert parsed_rec["subject_security_id"] == ""
    assert (parsed_rec["detailed_authentication_information_package_name_ntlm_only"] == "")
    assert parsed_rec["logon_type"] == ""
    assert parsed_rec["account_for_which_logon_failed_security_id"] == ""
    assert parsed_rec["detailed_authentication_information_key_length"] == ""
    assert parsed_rec["subject_logon_id"] == ""
    assert parsed_rec["process_information_caller_process_name"] == ""
    assert parsed_rec["process_information_caller_process_id"] == ""
    assert parsed_rec["subject_account_name"] == ""
    assert parsed_rec["process_information_process_name"] == ""
    assert parsed_rec["new_logon_account_name"] == ""
    assert parsed_rec["process_information_process_id"] == ""
    assert parsed_rec["failure_information_sub_status"] == ""
    assert parsed_rec["new_logon_security_id"] == ""
    assert parsed_rec["network_information_source_network_address"] == ""
    assert parsed_rec["detailed_authentication_information_transited_services"] == ""
    assert parsed_rec["new_logon_account_domain"] == ""
    assert parsed_rec["subject_account_domain"] == ""
    assert parsed_rec["detailed_authentication_information_logon_process"] == ""
    assert parsed_rec["account_for_which_logon_failed_account_domain"] == ""
    assert parsed_rec["account_for_which_logon_failed_account_name"] == ""
    assert parsed_rec["network_information_workstation_name"] == ""
    assert parsed_rec["network_information_source_port"] == "137"
    assert parsed_rec["application_information_process_id"] == "1048"
    assert (parsed_rec["application_information_application_name"] ==
            "\device\harddiskvolume1\windows\system32\svchost.exe")
    assert parsed_rec["network_information_direction"] == "inbound"
    assert parsed_rec["network_information_source_address"] == "100.20.100.30"
    assert parsed_rec["network_information_destination_address"] == "100.20.100.20"
    assert parsed_rec["network_information_destination_port"] == "137"
    assert parsed_rec["network_information_protocol"] == "0"
    assert parsed_rec["filter_information_filter_run_time_id"] == "65595"
    assert parsed_rec["filter_information_layer_name"] == "receive/accept"
    assert parsed_rec["filter_information_layer_run_time_id"] == "44"


def validate_4798(parsed_rec):
    assert parsed_rec["time"] == "04/03/2019 05:57:33 am"
    assert parsed_rec["id"] == "cf4876f3-716c-415c-994e-84acda054c9c"
    assert parsed_rec["eventcode"] == "4798"
    assert parsed_rec["subject_security_id"] == "null sid"
    assert parsed_rec["subject_account_name"] == ""
    assert parsed_rec["subject_logon_id"] == "0x0"
    assert parsed_rec["user_security_id"] == "null sid"
    assert parsed_rec["user_account_name"] == "hxyz"
    assert parsed_rec["user_account_domain"] == "hxyz-pc1"
    assert parsed_rec["process_information_process_id"] == "0x0"
    assert parsed_rec["process_information_process_name"] == "-"


def validate_4769(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 04:45:36 am"
    assert parsed_rec["id"] == "cf4876f3-716c-415c-994e-84acda054c9c"
    assert parsed_rec["eventcode"] == "4769"
    assert parsed_rec["account_information_account_name"] == "user@localhost.com"
    assert parsed_rec["account_information_account_domain"] == "localhost.com"
    assert (parsed_rec["account_information_logon_guid"] == "{1f1d4c09-e154-4898-4eb8-e3a03e130d11}")
    assert parsed_rec["service_information_service_name"] == "test.localhost.com"
    assert parsed_rec["service_information_service_id"] == "none_mapped"
    assert parsed_rec["network_information_client_address"] == "::ffff:100.10.100.20"
    assert parsed_rec["network_information_client_port"] == "26061"
    assert parsed_rec["additional_information_ticket_options"] == "0x40810000"
    assert parsed_rec["additional_information_ticket_encryption_type"] == "0x17"
    assert parsed_rec["additional_information_failure_code"] == "0x0"
    assert parsed_rec["additional_information_transited_services"] == "-"


def validate_4770(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 05:15:34 am"
    assert parsed_rec["id"] == "052b3a64-f1bd-4884-8e48-30b553bc495a"
    assert parsed_rec["eventcode"] == "4770"
    assert parsed_rec["account_information_account_name"] == "test@localhost.com"
    assert parsed_rec["account_information_account_domain"] == "localhost.com"
    assert parsed_rec["service_information_service_name"] == "user"
    assert parsed_rec["service_information_service_id"] == "localhost"
    assert parsed_rec["network_information_client_address"] == "::ffff:10.30.100.130"
    assert parsed_rec["network_information_client_port"] == "62133"
    assert parsed_rec["additional_information_ticket_options"] == "0x50800002"
    assert parsed_rec["additional_information_ticket_encryption_type"] == "0x12"


def validate_4771(parsed_rec):
    assert parsed_rec["time"] == "12/06/2018 06:52:05 am"
    assert parsed_rec["id"] == "cf4876f3-716c-415c-994e-84acda054c9c"
    assert parsed_rec["eventcode"] == "4771"
    assert parsed_rec["account_information_security_id"] == "localhost.com\lab"
    assert parsed_rec["account_information_account_name"] == "lab"
    assert parsed_rec["service_information_service_name"] == "user/localhost.com"
    assert parsed_rec["network_information_client_address"] == "100.20.1.70"
    assert parsed_rec["network_information_client_port"] == "60284"
    assert parsed_rec["additional_information_ticket_options"] == "0x40800000"
    assert parsed_rec["additional_information_failure_code"] == "0x18"
    assert parsed_rec["additional_information_pre_authentication_type"] == "2"
    assert parsed_rec["certificate_information_certificate_issuer_name"] == ""
    assert parsed_rec["certificate_information_certificate_serial_number"] == ""
    assert parsed_rec["certificate_information_certificate_thumbprint"] == ""


def validate_4781(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 05:15:34 am"
    assert parsed_rec["id"] == "052b3a64-f1bd-4884-8e48-30b553bc495a"
    assert parsed_rec["eventcode"] == "4781"
    assert parsed_rec["subject_security_id"] == "acme\\administrator"
    assert parsed_rec["subject_account_domain"] == "localhost.com"
    assert parsed_rec["subject_account_name"] == "test@localhost.com"
    assert parsed_rec["subject_logon_id"] == "0x1f40f"
    assert parsed_rec["target_account_security_id"] == "acme\\emp-nbonaparte"
    assert parsed_rec["target_account_account_domain"] == "acme"
    assert parsed_rec["target_account_old_account_name"] == "nbonaparte"
    assert parsed_rec["target_account_new_account_name"] == "emp-nbonaparte"
    assert parsed_rec["additional_information_privileges"] == "-"


def validate_4782(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 05:15:34 am"
    assert parsed_rec["id"] == "052b3a64-f1bd-4884-8e48-30b553bc495a"
    assert parsed_rec["eventcode"] == "4782"
    assert parsed_rec["subject_security_id"] == "acme\\administrator"
    assert parsed_rec["subject_account_domain"] == "localhost.com"
    assert parsed_rec["subject_account_name"] == "test@localhost.com"
    assert parsed_rec["subject_logon_id"] == "0x1f40f"
    assert parsed_rec["target_account_account_domain"] == "acme"
    assert parsed_rec["target_account_account_name"] == "nbonaparte"


def validate_4647(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 05:15:34 am"
    assert parsed_rec["id"] == "052b3a64-f1bd-4884-8e48-30b553bc495a"
    assert parsed_rec["eventcode"] == "4647"
    assert parsed_rec["subject_security_id"] == "anonymous logon"
    assert parsed_rec["subject_account_name"] == "appservice"
    assert parsed_rec["subject_account_domain"] == "domain001"
    assert parsed_rec["subject_logon_id"] == "0x27b9013"


def validate_4634(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 05:15:34 am"
    assert parsed_rec["id"] == "052b3a64-f1bd-4884-8e48-30b553bc495a"
    assert parsed_rec["eventcode"] == "4634"
    assert parsed_rec["subject_security_id"] == "anonymous logon"
    assert parsed_rec["subject_account_name"] == "appservice"
    assert parsed_rec["subject_account_domain"] == "domain001"
    assert parsed_rec["subject_logon_id"] == "0x27b9013"
    assert parsed_rec["logon_type"] == "3"


def validate_4648(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 05:15:34 am"
    assert parsed_rec["id"] == "052b3a64-f1bd-4884-8e48-30b553bc495a"
    assert parsed_rec["eventcode"] == "4648"
    assert parsed_rec["subject_account_name"] == "administrator"
    assert parsed_rec["subject_account_domain"] == "win-r9h529rio4y"
    assert parsed_rec["subject_logon_id"] == "0x1ba0e"
    assert parsed_rec["subject_logon_guid"] == "{00000000-0000-0000-0000-000000000000}"
    assert (parsed_rec["account_whose_credentials_were_used_account_name"] == "rsmith@mtg.com")
    assert (parsed_rec["account_whose_credentials_were_used_account_domain"] == "win-r9h529rio4y")
    assert (parsed_rec["account_whose_credentials_were_used_logon_guid"] == "{00000000-0000-0000-0000-000000000000}")
    assert parsed_rec["target_server_target_server_name"] == "sp01.icemail.com"
    assert parsed_rec["target_server_additional_information"] == "sp01.icemail.com"
    assert parsed_rec["process_information_process_id"] == "0x77c"
    assert (parsed_rec["process_information_process_name"] == "c:\program files\internet explorer\iexplore.exe")
    assert parsed_rec["network_information_network_address"] == "-"
    assert parsed_rec["network_information_port"] == "-"


def validate_4672(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 10:52:50 am"
    assert parsed_rec["id"] == "052b3a64-f1bd-4884-8e48-30b553bc495a"
    assert parsed_rec["eventcode"] == "4672"
    assert parsed_rec["time"] == "09/27/2018 10:52:50 am"
    assert parsed_rec["id"] == "052b3a64-f1bd-4884-8e48-30b553bc495a"
    assert parsed_rec["eventcode"] == "4672"
    assert parsed_rec["subject_security_id"] == "devuser"
    assert parsed_rec["subject_account_name"] == "user"
    assert parsed_rec["subject_account_domain"] == "dev"
    assert parsed_rec["subject_logon_id"] == "0x800a513d"
    assert (
        parsed_rec["privileges"] ==
        "sesecurityprivilege|sebackupprivilege|serestoreprivilege|setakeownershipprivilege|sedebugprivilege|sesystemenvironmentprivilege|seloaddriverprivilege|seimpersonateprivilege"
    )


def validate_4673(parsed_rec):
    assert parsed_rec["time"] == "04/30/2018 05:13:59 pm"
    assert parsed_rec["id"] == "sdgfhsdfhj-3245-dsf"
    assert parsed_rec["eventcode"] == "4673"
    assert parsed_rec["subject_security_id"] == "nt authority\\system"
    assert parsed_rec["subject_account_domain"] == "test.com"
    assert parsed_rec["subject_account_name"] == "fvjbvfjbvf$"
    assert parsed_rec["subject_logon_id"] == "0x3e7"
    assert (parsed_rec["service_server"] == "nt local security authority / authentication service")
    assert parsed_rec["service_service_name"] == "lsaregisterlogonprocess()"
    assert parsed_rec["process_process_id"] == "0x234"
    assert parsed_rec["process_process_name"] == "c:\windows\system32\lsass.exe"
    assert parsed_rec["privileges"] == "setcbprivilege"


def validate_4722(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 09:56:10 am"
    assert parsed_rec["id"] == "tdr5d-fjfgg-687bv-klhk"
    assert parsed_rec["eventcode"] == "4722"
    assert parsed_rec["subject_security_id"] == "test.com\\dhgfckkcg"
    assert parsed_rec["subject_account_domain"] == "test.com"
    assert parsed_rec["subject_account_name"] == "dhgfckkcg"
    assert parsed_rec["subject_logon_id"] == "0x2d55e5ef7"
    assert parsed_rec["target_account_security_id"] == "test.com\\hgcghjj"
    assert parsed_rec["target_account_account_domain"] == "test.com"
    assert parsed_rec["target_account_account_name"] == "hgcghjj"


def validate_4720(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 09:56:10 am"
    assert parsed_rec["id"] == "tdr5d-fjfgg-687bv-klhk"
    assert parsed_rec["eventcode"] == "4720"
    assert parsed_rec["subject_security_id"] == "acme-fr\administrator"
    assert parsed_rec["subject_account_domain"] == "acme-fr"
    assert parsed_rec["subject_account_name"] == "administrator"
    assert parsed_rec["subject_logon_id"] == "0x20f9d"
    assert parsed_rec["new_account_security_id"] == "acme-fr\john.lockeaccount"
    assert parsed_rec["new_account_account_name"] == "john.locke"
    assert parsed_rec["new_account_domain_name"] == "acme-fr"
    assert parsed_rec["attributes_sam_account_name"] == "john.locke"
    assert parsed_rec["attributes_display_name"] == "john locke"
    assert parsed_rec["attributes_user_principal_name"] == "john.locke@acme-fr.local"
    assert parsed_rec["attributes_home_directory"] == "-"
    assert parsed_rec["attributes_home_drive"] == "-"
    assert parsed_rec["attributes_script_path"] == "-"
    assert parsed_rec["attributes_profile_path"] == "-"
    assert parsed_rec["attributes_user_workstations"] == "-"
    assert parsed_rec["attributes_password_last_set"] == "<never>"
    assert parsed_rec["attributes_account_expires"] == "<never>"
    assert parsed_rec["attributes_primary_group_id"] == "513"
    assert parsed_rec["attributes_allowed_to_delegate_to"] == "-"
    assert parsed_rec["attributes_old_uac_value"] == "0x0"
    assert parsed_rec["attributes_new_uac_value"] == "0x15"
    assert (parsed_rec["attributes_user_account_control"] ==
            "account disabled|'password not required' - enabled|'normal account' - enable")
    assert parsed_rec["attributes_user_parameters"] == "-"
    assert parsed_rec["attributes_sid_history"] == "-"
    assert parsed_rec["attributes_logon_hours"] == "<value not set>"
    assert parsed_rec["additional_information_privileges"] == "-"


def validate_4723(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 10:24:34 am"
    assert parsed_rec["id"] == "tdr5d-fjfgg-687bv-klhk"
    assert parsed_rec["eventcode"] == "4723"
    assert parsed_rec["subject_security_id"] == "test.com\\auser"
    assert parsed_rec["subject_account_domain"] == "test.com"
    assert parsed_rec["subject_account_name"] == "auser"
    assert parsed_rec["subject_logon_id"] == "0x258440926"
    assert parsed_rec["target_account_security_id"] == "test.com\\auser"
    assert parsed_rec["target_account_account_domain"] == "test.com"
    assert parsed_rec["target_account_account_name"] == "auser"
    assert parsed_rec["additional_information_privileges"] == "-"


def validate_4724(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 10:24:34 am"
    assert parsed_rec["id"] == "tdr5d-fjfgg-687bv-klhk"
    assert parsed_rec["eventcode"] == "4724"
    assert parsed_rec["subject_security_id"] == "test.com\\auser"
    assert parsed_rec["subject_account_domain"] == "test.com"
    assert parsed_rec["subject_account_name"] == "auser"
    assert parsed_rec["subject_logon_id"] == "0x258440926"
    assert parsed_rec["target_account_security_id"] == "test.com\\auser"
    assert parsed_rec["target_account_account_domain"] == "test.com"
    assert parsed_rec["target_account_account_name"] == "auser"


def validate_4725(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 10:24:34 am"
    assert parsed_rec["id"] == "tdr5d-fjfgg-687bv-klhk"
    assert parsed_rec["eventcode"] == "4725"
    assert parsed_rec["subject_security_id"] == "test.com\\auser"
    assert parsed_rec["subject_account_domain"] == "test.com"
    assert parsed_rec["subject_account_name"] == "auser"
    assert parsed_rec["subject_logon_id"] == "0x258440926"
    assert parsed_rec["target_account_security_id"] == "test.com\\auser"
    assert parsed_rec["target_account_account_domain"] == "test.com"
    assert parsed_rec["target_account_account_name"] == "auser"


def validate_4726(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 10:24:34 am"
    assert parsed_rec["id"] == "tdr5d-fjfgg-687bv-klhk"
    assert parsed_rec["eventcode"] == "4726"
    assert parsed_rec["subject_security_id"] == "test.com\\auser"
    assert parsed_rec["subject_account_domain"] == "test.com"
    assert parsed_rec["subject_account_name"] == "auser"
    assert parsed_rec["subject_logon_id"] == "0x258440926"
    assert parsed_rec["target_account_security_id"] == "test.com\\auser"
    assert parsed_rec["target_account_account_domain"] == "test.com"
    assert parsed_rec["target_account_account_name"] == "auser"
    assert parsed_rec["additional_information_privileges"] == "-"


def validate_4732(parsed_rec):
    assert parsed_rec["time"] == "09/19/2018 06:18:24 am"
    assert parsed_rec["id"] == "tdr5d-fjfgg-687bv-klhk"
    assert parsed_rec["eventcode"] == "4732"
    assert parsed_rec["subject_security_id"] == "nt authority\system"
    assert parsed_rec["subject_account_domain"] == "localhost.com"
    assert parsed_rec["subject_account_name"] == "testuser$"
    assert parsed_rec["subject_logon_id"] == "0x3e7"
    assert parsed_rec["member_security_id"] == "testuser\offer"
    assert parsed_rec["member_account_name"] == "-"
    assert parsed_rec["group_security_id"] == "testuser\offer remote assistance helpers"
    assert parsed_rec["group_group_name"] == "offer remote assistance helpers"
    assert parsed_rec["group_group_domain"] == "testuser"
    assert parsed_rec["additional_information_privileges"] == "-"


def validate_4738(parsed_rec):
    assert parsed_rec["time"] == "05/01/2018 05:41:37 am"
    assert parsed_rec["id"] == "tdr5d-fjfgg-687bv-klhk"
    assert parsed_rec["eventcode"] == "4738"
    assert parsed_rec["subject_security_id"] == "nt authority\system"
    assert parsed_rec["subject_account_domain"] == "prod"
    assert parsed_rec["subject_account_name"] == "esfdhf06$"
    assert parsed_rec["subject_logon_id"] == "0x3e7"
    assert parsed_rec["target_account_security_id"] == "esfdhf06\\atble"
    assert parsed_rec["target_account_account_domain"] == "esfdhf06"
    assert parsed_rec["target_account_account_name"] == "atble"
    assert parsed_rec["additional_information_privileges"] == "-"
    assert parsed_rec["changed_attributes_sam_account_name"] == "atble"
    assert parsed_rec["changed_attributes_home_directory"] == "<value not set>"
    assert parsed_rec["changed_attributes_primary_group_id"] == "513"
    assert parsed_rec["changed_attributes_user_principal_name"] == "-"
    assert parsed_rec["changed_attributes_profile_path"] == "<value not set>"
    assert parsed_rec["changed_attributes_user_workstations"] == "<value not set>"
    assert parsed_rec["changed_attributes_user_parameters"] == "-"
    assert parsed_rec["changed_attributes_script_path"] == "<value not set>"
    assert parsed_rec["changed_attributes_display_name"] == "mike atble"
    assert parsed_rec["changed_attributes_home_drive"] == "<value not set>"
    assert parsed_rec["changed_attributes_new_uac_value"] == "0x210"
    assert parsed_rec["changed_attributes_logon_hours"] == "all"
    assert parsed_rec["changed_attributes_account_expires"] == "<never>"
    assert parsed_rec["changed_attributes_old_uac_value"] == "0x210"
    assert parsed_rec["changed_attributes_password_last_set"] == "5/1/2018 5:41:37 am"
    assert parsed_rec["changed_attributes_allowedtodelegateto"] == "-"
    assert parsed_rec["changed_attributes_user_account_control"] == "-"
    assert parsed_rec["changed_attributes_sid_history"] == "-"


def validate_4740(parsed_rec):
    assert parsed_rec["time"] == "09/28/2018 01:53:37 am"
    assert parsed_rec["id"] == "tdr5d-fjfgg-687bv-klhk"
    assert parsed_rec["eventcode"] == "4740"
    assert parsed_rec["subject_security_id"] == "nt authority\system"
    assert parsed_rec["subject_account_domain"] == "nvdmz"
    assert parsed_rec["subject_account_name"] == "sdgbjsd02$"
    assert parsed_rec["subject_logon_id"] == "0x3e7"
    assert parsed_rec["account_locked_out_security_id"] == "sdgbjsd02\guest"
    assert parsed_rec["account_locked_out_account_name"] == "guest"
    assert parsed_rec["additional_information_caller_computer_name"] == "sdgbjsd01"


def validate_4743(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 10:24:34 am"
    assert parsed_rec["id"] == "tdr5d-fjfgg-687bv-klhk"
    assert parsed_rec["eventcode"] == "4743"
    assert parsed_rec["subject_security_id"] == "test.com\\auser"
    assert parsed_rec["subject_account_domain"] == "test.com"
    assert parsed_rec["subject_account_name"] == "auser"
    assert parsed_rec["subject_logon_id"] == "0x258440926"
    assert parsed_rec["target_account_security_id"] == "test.com\\auser"
    assert parsed_rec["target_account_account_domain"] == "test.com"
    assert parsed_rec["target_account_account_name"] == "auser"
    assert parsed_rec["additional_information_privileges"] == "-"


def validate_4756(parsed_rec):
    assert parsed_rec["time"] == "09/19/2018 06:18:24 am"
    assert parsed_rec["id"] == "tdr5d-fjfgg-687bv-klhk"
    assert parsed_rec["eventcode"] == "4756"
    assert parsed_rec["subject_security_id"] == "nt authority\\system"
    assert parsed_rec["subject_account_domain"] == "localhost.com"
    assert parsed_rec["subject_account_name"] == "testuser$"
    assert parsed_rec["subject_logon_id"] == "0x3e7"
    assert parsed_rec["member_security_id"] == "testuser\offer"
    assert parsed_rec["member_account_name"] == "-"
    assert parsed_rec["group_security_id"] == "testuser\offer remote assistance helpers"
    assert parsed_rec["group_group_name"] == "offer remote assistance helpers"
    assert parsed_rec["group_group_domain"] == "testuser"
    assert parsed_rec["additional_information_privileges"] == "-"


def validate_4767(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 10:24:34 am"
    assert parsed_rec["id"] == "tdr5d-fjfgg-687bv-klhk"
    assert parsed_rec["eventcode"] == "4767"
    assert parsed_rec["subject_security_id"] == "test.com\\auser"
    assert parsed_rec["subject_account_domain"] == "test.com"
    assert parsed_rec["subject_account_name"] == "auser"
    assert parsed_rec["subject_logon_id"] == "0x258440926"
    assert parsed_rec["target_account_security_id"] == "test.com\\auser"
    assert parsed_rec["target_account_account_domain"] == "test.com"
    assert parsed_rec["target_account_account_name"] == "auser"


def validate_4768(parsed_rec):
    assert parsed_rec["time"] == "09/27/2018 09:08:02 am"
    assert parsed_rec["id"] == "asd-eter-34235-fgd-346"
    assert parsed_rec["eventcode"] == "4768"
    assert parsed_rec["network_information_client_address"] == "::ffff:10.20.90.30"
    assert parsed_rec["network_information_client_port"] == "6349"
    assert parsed_rec["service_information_service_name"] == "asdfgrvk"
    assert parsed_rec["service_information_service_id"] == "localhost.com\\asdfgrvk"
    assert parsed_rec["account_information_account_name"] == "healthmailbox06ca30c"
    assert parsed_rec["account_information_supplied_realm_name"] == "localhost.com"
    assert (parsed_rec["account_information_user_id"] == "localhost.com\healthmailbox06ca30c")
    assert parsed_rec["additional_information_result_code"] == "0x0"
    assert parsed_rec["additional_information_ticket_options"] == "0x40810010"
    assert parsed_rec["additional_information_ticket_encryption_type"] == "0x12"
    assert parsed_rec["additional_information_pre_authentication_type"] == "2"
    assert parsed_rec["certificate_information_certificate_issuer_name"] == ""
    assert parsed_rec["certificate_information_certificate_serial_number"] == ""
    assert parsed_rec["certificate_information_certificate_thumbprint"] == ""


def unknown_record_type(parsed_rec):
    raise Exception("Unknown eventcode appeared")


VALIDATE_DICT = {
    "4624": validate_4624,
    "4625": validate_4625,
    "4634": validate_4634,
    "4647": validate_4647,
    "4648": validate_4648,
    "4672": validate_4672,
    "4673": validate_4673,
    "4720": validate_4720,
    "4722": validate_4722,
    "4723": validate_4723,
    "4724": validate_4724,
    "4725": validate_4725,
    "4726": validate_4726,
    "4732": validate_4732,
    "4738": validate_4738,
    "4740": validate_4740,
    "4743": validate_4743,
    "4756": validate_4756,
    "4767": validate_4767,
    "4768": validate_4768,
    "4769": validate_4769,
    "4770": validate_4770,
    "4771": validate_4771,
    "4781": validate_4781,
    "4782": validate_4782,
    "4798": validate_4798,
    "5156": validate_5156,
    "5157": validate_5157,
}


def test_windows_event_parser():
    wep = WindowsEventParser()
    test_input_df = cudf.DataFrame()
    raw_colname = "_raw"
    test_input_df[raw_colname] = TEST_DATA
    test_output_df = wep.parse(test_input_df, raw_colname)
    for parsed_rec in test_output_df.to_records():
        eventcode = parsed_rec["eventcode"]
        validate_func = VALIDATE_DICT.get(eventcode, unknown_record_type)
        validate_func(parsed_rec)


def test2_windows_event_parser():
    wep = WindowsEventParser(interested_eventcodes=["5156"])
    test_input_df = cudf.DataFrame()
    raw_colname = "_raw"
    test_input_df[raw_colname] = TEST_DATA
    test_output_df = wep.parse(test_input_df, raw_colname)
    parsed_rec = test_output_df.to_records()[0]
    assert parsed_rec["time"] == "04/03/2019 11:58:59 am"
    assert parsed_rec["id"] == "c3f48bba-90a1-4999-84a6-4da9d964d31d"
    assert parsed_rec["eventcode"] == "5156"
    assert parsed_rec["application_information_process_id"] == "4"
    assert parsed_rec["application_information_application_name"] == "system"
    assert parsed_rec["network_information_direction"] == "inbound"
    assert parsed_rec["network_information_source_address"] == "100.20.100.20"
    assert parsed_rec["network_information_source_port"] == "138"
    assert parsed_rec["network_information_destination_address"] == "100.20.100.30"
    assert parsed_rec["network_information_destination_port"] == "138"
    assert parsed_rec["network_information_protocol"] == "17"
    assert parsed_rec["filter_information_filter_run_time_id"] == "0"
    assert parsed_rec["filter_information_layer_name"] == "receive/accept"
    assert parsed_rec["filter_information_layer_run_time_id"] == "44"


def test3_windows_event_parser():
    expected_error = KeyError(
        "Regex for eventcode 24 is not available in the config file. Please choose from ['4624', '4625', '4634', '4647', '4648', '4672', '4673', '4720', '4722', '4723', '4724', '4725', '4726', '4732', '4738', '4740', '4743', '4756', '4767', '4768', '4769', '4770', '4771', '4781', '4782', '4798', '5156', '5157']"
    )
    with pytest.raises(KeyError) as actual_error:
        WindowsEventParser(interested_eventcodes=["5156", "24"])
        assert actual_error == expected_error
