import ssl
from email.header import Header
from email.mime.text import MIMEText
from email.message import EmailMessage
import smtplib


def send_myself_QQEmail(title: str, content: str):
    EMAIL_ADDRESS = '1178890320@qq.com'
    EMAIL_PASSWORD = 'sfsrautfqvqobacg'

    context = ssl.create_default_context()
    sender = EMAIL_ADDRESS
    receiver = EMAIL_ADDRESS

    msg = EmailMessage()
    msg['subject'] = title
    msg['From'] = sender
    msg['To'] = receiver
    msg.set_content(content)

    try:
        with smtplib.SMTP_SSL("smtp.qq.com", 465, context=context) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print('Successfully send email to 1178890320@qq.com')
    except OSError as e:
        print('Failed send email to 1178890320@qq.com')
        print(e)
