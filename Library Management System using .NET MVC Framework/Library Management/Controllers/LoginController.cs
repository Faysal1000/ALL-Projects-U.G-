using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Mail;
using System.Net;
using System.Web;
using System.Web.Mvc;
using Library_Management.Models;
using System.Data.Entity;

namespace Library_Management.Controllers
{
    public class LoginController : Controller
    {
        private LibraryDBEntities1 db;
        public  LoginController()
        {
            db = new LibraryDBEntities1();
        }

        [HttpGet]
        public ActionResult Login()
        {

            return View();
        }

        [HttpPost]
        public ActionResult Login(User data)
        {

            var user = db.Users.FirstOrDefault(u => u.Email == data.Email && u.Password == data.Password);
            if (user != null)
            {
                Session["UserId"] = user.ID;
                Session["UserRole"] = user.Role;
                Session["UserName"] = user.Name;

                if (user.Role == "Admin")
                {
                    SendOverdueReminderEmails();
                    return RedirectToAction("Dashboard", "Admin");
                }
                return RedirectToAction("Books", "Member");
            }
            TempData["msg"] = "Email or Password is Wrong!";

            return View(data);
        }

        public ActionResult Logout()
        {
            Session.Clear();
            TempData["SuccessMessage"] = "Logged Out Successfully";
            return RedirectToAction("Login");
        }
        public void SendOverdueReminderEmails()
        {
            var overdueList = db.BorrowHistories
                                .Where(b => b.ReturnedAt == null &&
                                            b.BorrowedAt != null &&
                                            DbFunctions.AddDays(b.BorrowedAt.Value, 0) < DateTime.Now)
                                .ToList();

            foreach (var entry in overdueList)
            {
                string userEmail = entry.User.Email;
                string userName = entry.User.Name;
                string bookTitle = entry.Book.Title;
                DateTime borrowedDate = entry.BorrowedAt.Value;

                string subject = "⏰ Book Return Reminder";
                string message = $@"
                                <p>Dear 
                                    {userName},</p>
                                <p>This is a friendly reminder that you borrowed the book <strong>{bookTitle}</strong> on <strong>{borrowedDate.ToString("MMMM dd, yyyy")}</strong>.</p>
                                <p> According to our policy, books must be returned within 7 days.Your return is now overdue.</p>
                                <p> Please return the book as soon as possible to avoid any penalties or restrictions on future borrowing.</p>
                                <br/>
                                <p> Thank you for your cooperation.</p>
                                <p> Best regards,<br/> Library Management Team </p>";

                sendmail(userEmail, message, subject);

            }

        }
        public string sendmail(string mailAddress, string message, string subject)
        {

            var smtpClient = new SmtpClient("smtp.gmail.com")
            {
                Port = 587,
                Credentials = new NetworkCredential("asefrahman2001@gmail.com", "buqjdehvvvegbjgs"),
                EnableSsl = true,
            };
            var mailMessage = new MailMessage
            {
                From = new MailAddress("asefrahman2001@gmail.com"),
                Subject = subject,
                Body = message,
                IsBodyHtml = true
            };
            mailMessage.To.Add(mailAddress);
            try
            {
                smtpClient.Send(mailMessage);
                return "Mailed";
            }
            catch (Exception ex)
            {
                return $"Error sending email: {ex.Message}";
            }
        }

    }
}