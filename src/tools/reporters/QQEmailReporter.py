from email.header import Header
from email.mime.text import MIMEText
import smtplib


def send_myself_QQEmail(title: str, content: str):
    # 发件人邮箱
    sender = '1178890320@qq.com'
    # 收件人邮箱
    receiver = '1178890320@qq.com'

    # 发送邮件的服务器
    smtp_server = 'smtp.qq.com'
    # 发件人邮箱的密码或授权码
    password = 'wnskzqmvldzcjebg'

    # 创建邮件对象
    message = MIMEText(content, 'plain', 'utf-8')
    message['From'] = Header(sender, 'utf-8')
    message['To'] = Header(receiver, 'utf-8')
    message['Subject'] = Header(title, 'utf-8')

    # 发送邮件
    try:
        smtpObj = smtplib.SMTP_SSL(smtp_server, 465)
        smtpObj.login(sender, password)
        smtpObj.sendmail(sender, receiver, message.as_string())
        print('Successfully send email to 1178890320@qq.com')
    except smtplib.SMTPException as e:
        print('Failed send email to 1178890320@qq.com')
        print(e)