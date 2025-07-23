using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Mail;
using System.Net;
using System.Web;
using System.Xml.Linq;
using System.Drawing;
using System.Security.Policy;

namespace LifeinnovirorMentalHealthConsultency.Functional_Class
{
    public static class EmailManagement
    {
        private const string sender = "faysalahmmed4200@gmail.com"; 
        public static string AccountCreationMail(string name, string receiver)
        {
            string message = $@"
                    <html>
                    <body style='font-family: Arial, sans-serif; line-height: 1.6;'>
                        <p>Dear <strong>{name}</strong>,</p>

                        <p>Thank you for creating an account on <strong>Lifeinnoviror</strong>. We are excited to have you on board!</p>

                        <p>You can now log in to your profile and explore our services.</p>

                        <p>
                            <a href='https://lifeinnoviror.com/login' style='color: #2a8bf2; font-weight: bold;'>
                                Click here to login https://lifeinnoviror.com/login
                            </a>
                        </p>

                        <p>If you have any questions or need assistance, feel free to contact our support team.</p>

                        <p>Best regards,<br />
                        <strong>Faysal Ahmmed</strong><br />
                        Lifeinnoviror Team</p>
                    </body>
                    </html>
            ";

            string subject = "Welcome to Lifeinnoviror – Your Account Has Been Successfully Created";

            return SendEmail(sender, receiver, subject, message);
        }




        public static string PatientAutoAccountCreationMail(string name, string email)
        {
            string message = $@"
                    <html>
                    <body style='font-family: Arial, sans-serif; line-height: 1.6;'>
                        <p>Dear <strong>{name}</strong>,</p>

                        <p>Your account has been successfully created on our Lifeinnoviror system. You can now log in and access your profile and manage appointment services.</p>

                        <p><strong>Login Credentials:</strong></p>
                        <ul style='list-style-type: none; padding-left: 0;'>
                            <li><strong>Email:</strong> <span style='color: #2a8bf2;'>{email}</span></li>
                            <li><strong>Password:</strong> <span style='color: #f23c2a;'>{email}</span></li>
                        </ul>

                        <p>You can log in using the following link:<br />
                        <a href='https://lifeinnoviror.com/login' style='color: #2a8bf2;'>https://lifeinnoviror.com/login</a></p>

                        <p>Please make sure to change your password after your first login for security purposes.</p>
                        
                        <p>If you have any questions or need assistance, feel free to contact our support team.</p>

                        <p>Best regards,<br />
                        <strong>Faysal Ahmmed</strong><br />
                        Lifeinnoviror Team</p>
                    </body>
                    </html>";

            string subject = "Welcome to Lifeinnoviror – Your Account Has Been Successfully Created by our System";

            return SendEmail(sender, email, subject, message);
        }





        private static string SendEmail(string sender, string receiver, string subject, string message)
        {
            var smtpClient = new SmtpClient("smtp.gmail.com")
            {
                Port = 587,
                Credentials = new NetworkCredential("faysalahmmed4200@gmail.com", "ncxaxuhsnaqrutnr"),
                EnableSsl = true,
            };
            var mailMessage = new MailMessage
            {
                From = new MailAddress(sender),
                Subject = subject,
                Body = message,
                IsBodyHtml = true
            };
            mailMessage.To.Add(receiver);
            try
            {
                smtpClient.Send(mailMessage);
                return "Mail Sent!";
            }
            catch (Exception ex)
            {
                return $"Error sending email: {ex.Message}";
            }
        }
    }
}