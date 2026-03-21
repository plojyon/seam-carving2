#!/usr/bin/expect -f

set timeout 10
set otp [exec oathtool --totp -b RQEEWCDCFHHJFKDWFA7JXEYRGQ]

spawn ssh hpc
expect "Verification code:"
send "$otp\r"

interact
