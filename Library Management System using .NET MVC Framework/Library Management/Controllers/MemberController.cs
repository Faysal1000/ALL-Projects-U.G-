using System;
using System.Collections.Generic;
using System.Data.Entity;
using System.Linq;
using System.Net.Mail;
using System.Net;
using System.Web;
using System.Web.Mvc;
using Library_Management.Classes;
using Library_Management.Models;

namespace Library_Management.Controllers
{
    public class MemberController : Controller
    {
        LibraryDBEntities1 db ;
        
        public MemberController()
        {
            db = new LibraryDBEntities1();
        }

        [HttpGet]
        public ActionResult Register()
        {
            return View();
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Register(User model, string confirm_password)
        {
            if (!ModelState.IsValid)
            {
                return View(model);
            }
            if (confirm_password != model.Password)
            {
                ModelState.AddModelError("Password", "Password Didn't Match");
                return View(model);
            }

            // Check if email already exists
            bool emailExists = db.Users.Any(u => u.Email.ToLower() == model.Email.ToLower());
            if (emailExists)
            {
                ModelState.AddModelError("Email", "Email is already registered.");
                return View(model);
            }

            model.Role = "Member";
            model.CreatedAt = DateTime.Now;

            db.Users.Add(model);
            db.SaveChanges();

            TempData["SuccessMessage"] = "Registration successful. You can now log in.";
            return RedirectToAction("Login", "Login"); 
        }



        [HttpGet]
        public ActionResult Books(string searchTerm)
        {
            var books = db.Books
                         .Where(b => b.Quantity >= 0);

            if (!string.IsNullOrWhiteSpace(searchTerm))
            {
                string term = searchTerm.ToLower();
                books = books.Where(b =>
                    b.Title.ToLower().Contains(term) ||
                    b.Author.ToLower().Contains(term));
            }

            return View(books.ToList());
        }

        [HttpGet]
        public ActionResult BookDetails(int id)
        {
            int? userId = Session["UserId"] as int?;
            if (userId == null)
                return RedirectToAction("Login", "Login");

            var book = db.Books.Find(id);
            if (book == null)
                return HttpNotFound();

            return View(book);
        }


        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult RequestBorrow(int bookId)
        {
            int? memberId = Session["UserId"] as int?;
            if (memberId == null)
                return RedirectToAction("Login", "Login");

            //  Check if member already borrowed the book and has not returned it
            bool alreadyBorrowed = db.BorrowHistories.Any(h =>
                h.MemberId == memberId &&
                h.BookId == bookId &&
                h.ReturnedAt == null);

            if (alreadyBorrowed)
            {
                TempData["ErrorMessage"] = "You have already borrowed this book and haven't returned it yet.";
                return RedirectToAction("BookDetails", new { id = bookId });
            }

            //  Check if there's already a pending borrow request
            bool alreadyRequested = db.BorrowRequests.Any(r =>
                r.MemberId == memberId &&
                r.BookId == bookId &&
                !r.ApprovedAt.HasValue);

            if (alreadyRequested)
            {
                TempData["ErrorMessage"] = "You have already requested this book.";
                return RedirectToAction("BookDetails", new { id = bookId });
            }

            //  Add borrow request
            var request = new BorrowRequest
            {
                MemberId = memberId.Value,
                BookId = bookId,
                Status = false,
                RequestedAt = DateTime.Now
            };

            db.BorrowRequests.Add(request);
            db.SaveChanges();

            TempData["SuccessMessage"] = "Borrow request sent successfully.";
            return RedirectToAction("BookDetails", new { id = bookId });
        }


        [HttpGet]
        public ActionResult MyBorrowOverview()
        {
            int? memberId = Session["UserId"] as int?;
            if (memberId == null)
                return RedirectToAction("Login", "Login");

            var model = new MemberBorrowOverview
            {
                BorrowedBooks = db.BorrowHistories
                                  .Where(h => h.MemberId == memberId)
                                  .Include("Book")
                                  .OrderByDescending(h => h.BorrowedAt)
                                  .ToList(),

                BorrowRequests = db.BorrowRequests
                                   .Where(r => r.MemberId == memberId && r.ApprovedAt == null)
                                   .Include("Book")
                                   .OrderByDescending(r => r.RequestedAt)
                                   .ToList()
            };

            return View(model);
        }


        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult RequestReturn(int id)
        {
            int? memberId = Session["UserId"] as int?;
            if (memberId == null)
                return RedirectToAction("Login", "Login");

            var history = db.BorrowHistories.FirstOrDefault(h => h.ID == id && h.MemberId == memberId);
            if (history == null || history.ReturnedAt != null)
            {
                TempData["ErrorMessage"] = "Invalid return request.";
                return RedirectToAction("MyBorrowOverview");
            }

            history.ReturnRequested = true;
            db.SaveChanges();

            TempData["SuccessMessage"] = "Return request submitted.";
            return RedirectToAction("MyBorrowOverview");
        }

        [HttpGet]
        public ActionResult ViewProfile()
        {
            int? userId = Session["UserId"] as int?;
            if (userId == null)
                return RedirectToAction("Login", "Login");

            var user = db.Users.FirstOrDefault(u => u.ID == userId && u.Role == "Member");
            if (user == null)
                return HttpNotFound();

            return View(user);
        }

        [HttpGet]
        public ActionResult EditProfile()
        {
            int? userId = Session["UserId"] as int?;
            if (userId == null)
                return RedirectToAction("Login", "Login");

            var user = db.Users.FirstOrDefault(u => u.ID == userId && u.Role == "Member");
            if (user == null)
                return HttpNotFound();

            return View(user);
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult EditProfile(User updatedUser, string confirm_password)
        {
            var member = db.Users.FirstOrDefault(u => u.ID == updatedUser.ID && u.Role == "Member");
            if (member == null)
                return HttpNotFound();

            // Only update password if filled
            if (!string.IsNullOrWhiteSpace(updatedUser.Password))
            {
                if (updatedUser.Password != confirm_password)
                {
                    ModelState.AddModelError("", "Password and Confirm Password do not match.");
                    return View(updatedUser);
                }

                member.Password = updatedUser.Password;
            }
            if (!ModelState.IsValid)
            {
                return View(updatedUser);
            }

            // Update fields
            member.Name = updatedUser.Name;
            member.Email = updatedUser.Email;
            member.Address = updatedUser.Address;
            db.SaveChanges();

            TempData["SuccessMessage"] = "Profile updated successfully!";
            return RedirectToAction("ViewProfile");

        }

        [HttpGet]
        public ActionResult ForgetPassword()
        {
            return View();
        }


        // Store OTP in TempData or Session (better to use Session)
        [HttpPost]
        public ActionResult SendCodeInMail(string mail)
        {
            var user = db.Users.FirstOrDefault(u => u.Email == mail && u.Role == "Member");
            if (user == null)
            {
                TempData["ErrorMessage"] = "Email not found!";
                return RedirectToAction("ForgetPassword");
            }

            // Generate code
            string code = new Random().Next(100000, 999999).ToString();
            Session["ResetCode"] = code;
            Session["ResetEmail"] = mail;

            string message = $"Your one-time password reset code is: <b>{code}</b>";

            var smtpClient = new SmtpClient("smtp.gmail.com")
            {
                Port = 587,
                Credentials = new NetworkCredential("asefrahman2001@gmail.com", "buqjdehvvvegbjgs"),
                EnableSsl = true,
            };
            var mailMessage = new MailMessage
            {
                From = new MailAddress("asefrahman2001@gmail.com"),
                Subject = "Password Reset Code",
                Body = message,
                IsBodyHtml = true
            };
            mailMessage.To.Add(mail);
            try
            {
                smtpClient.Send(mailMessage);
                return RedirectToAction("VerifyCode");
            }
            catch (Exception ex)
            {
                TempData["ErrorMessage"] = $"Error sending email: {ex.Message}";
                return RedirectToAction("ForgetPassword");
            }
        }

        [HttpGet]
        public ActionResult VerifyCode()
        {
            return View();
        }

        [HttpPost]
        public ActionResult VerifyCode(string code)
        {
            string storedCode = Session["ResetCode"]?.ToString();
            if (storedCode == code)
            {
                // Show password (NOT recommended — better to redirect to reset page)
                var resetMail = Session["ResetEmail"].ToString();
                var user = db.Users.FirstOrDefault(u => u.Email == resetMail && u.Role == "Member");
                TempData["SuccessMessage"] = $"Your password is: {user.Password}";
                return RedirectToAction("Login", "Login");
            }

            TempData["ErrorMessage"] = "Invalid code.";
            return View();
        }

    }
}