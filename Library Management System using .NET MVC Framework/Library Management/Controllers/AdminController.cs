using System;
using System.Collections.Generic;
using System.Data.Entity;
using System.IO;
using System.Linq;
using System.Net.Mail;
using System.Net;
using System.Web;
using System.Web.Mvc;
using Library_Management.Authorization;
using Library_Management.Classes;
using Library_Management.Models;
using System.Security.Policy;
using System.Data.Entity.Core.Common.CommandTrees.ExpressionBuilder;
using System.Net.Http;
using System.Runtime.Remoting.Lifetime;
using System.Threading.Tasks;

namespace Library_Management.Controllers
{
    [AdminOnly]
    public class AdminController : Controller
    {
        LibraryDBEntities1 db;

        public AdminController ()
        {
            db = new LibraryDBEntities1();
        }

        [HttpGet]
        public ActionResult Dashboard()
        {

            var today = DateTime.Today;
            var totalBorrowed = db.BorrowHistories.Count(b => b.ReturnedAt == null);
            int totalStock = db.Books.Sum(b => (int?)b.Quantity) ?? 0;

            var borrowedLast7Days = Enumerable.Range(0, 7)
                .Select(i => today.AddDays(-i))
                .OrderBy(d => d)
                .Select(d => new DailyStat
                {
                    DateLabel = d.ToString("MMM dd"),
                    Count = db.BorrowHistories.Count(b => DbFunctions.TruncateTime(b.BorrowedAt) == d)
                }).ToList();

            var returnedLast7Days = Enumerable.Range(0, 7)
                .Select(i => today.AddDays(-i))
                .OrderBy(d => d)
                .Select(d => new DailyStat
                {
                    DateLabel = d.ToString("MMM dd"),
                    Count = db.BorrowHistories.Count(b => DbFunctions.TruncateTime(b.ReturnedAt) == d)
                }).ToList();

            var model = new AdminDashboardViewModel
            {
                TotalBorrowed = totalBorrowed,
                TotalInStock = totalStock,
                BorrowedLast7Days = borrowedLast7Days,
                ReturnedLast7Days = returnedLast7Days
            };

            return View(model);
        }




        [HttpGet]
        public ActionResult AllMembers()
        {
            var members = db.Users
                            .Where(u => u.Role == "Member")
                            .ToList();

            return View(members);
        }

        [HttpGet]
        public ActionResult ViewMember(int id)
        {
            var member = db.Users.FirstOrDefault(u => u.ID == id && u.Role == "Member");
            if (member == null)
            {
                return HttpNotFound();
            }
            return View(member);
        }

        [HttpGet]
        public ActionResult AddMember()
        {
            return View();
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult AddMember(User model)
        {
            if (!ModelState.IsValid)
            {
                return View(model);
            }

            // Check for email uniqueness
            bool emailExists = db.Users.Any(u => u.Email.ToLower() == model.Email.ToLower());
            if (emailExists)
            {
                ModelState.AddModelError("Email", "This email is already registered.");
                return View(model);
            }

            model.Role = "Member";
            model.CreatedAt = DateTime.Now;

            db.Users.Add(model);
            db.SaveChanges();

            TempData["SuccessMessage"] = "Member added successfully.";
            return RedirectToAction("AllMembers");
        }

        [HttpGet]
        public ActionResult EditMember(int id)
        {
            var member = db.Users.FirstOrDefault(u => u.ID == id && u.Role == "Member");
            if (member == null)
            {
                return HttpNotFound();
            }
            return View(member);
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult EditMember(User updatedUser)
        {

            var member = db.Users.FirstOrDefault(u => u.ID == updatedUser.ID && u.Role == "Member");
            if (member == null)
                return HttpNotFound();

            // Only update password if filled
            if (!string.IsNullOrWhiteSpace(updatedUser.Password))
            {
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
            return RedirectToAction("AllMembers");

        }
        [HttpGet]
        public ActionResult DeleteMember(int id)
        {
            var member = db.Users.FirstOrDefault(u => u.ID == id && u.Role == "Member");
            if (member == null)
                return HttpNotFound();

            db.Users.Remove(member);
            db.SaveChanges();
            return RedirectToAction("AllMembers");
        }



        [HttpGet]
        public ActionResult AllBooks()
        {
            var books = db.Books.ToList();
            return View(books);
        }

        [HttpGet]
        public ActionResult ViewBook(int id)
        {
            var book = db.Books.FirstOrDefault(b => b.ID == id);
            if (book == null)
                return HttpNotFound();

            return View(book);
        }

        [HttpGet]
        public ActionResult EditBook(int id)
        {
            var book = db.Books.Find(id);
            if (book == null)
                return HttpNotFound();

            return View(book);
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult EditBook(Book model, HttpPostedFileBase CoverImage)
        {
            if (!ModelState.IsValid)
            {
                return View(model);
            }

            var book = db.Books.Find(model.ID);
            if (book == null)
                return HttpNotFound();

            // Update basic fields
            book.Title = model.Title.Trim();
            book.Author = model.Author.Trim();
            book.Category = model.Category.Trim();
            book.Quantity = model.Quantity;
            book.Description = model.Description;

            // Handle image upload (optional)
            if (CoverImage != null && CoverImage.ContentLength > 0)
            {
                string extension = Path.GetExtension(CoverImage.FileName);
                string fileName = book.ID + extension;
                string folderPath = Server.MapPath("~/BookImg/");
                string fullPath = Path.Combine(folderPath, fileName);

                // Ensure folder exists
                if (!Directory.Exists(folderPath))
                {
                    Directory.CreateDirectory(folderPath);
                }

                // Overwrite old image
                CoverImage.SaveAs(fullPath);

                // Update image URL
                book.CoverImageUrl = "~/BookImg/" + fileName;
            }

            db.SaveChanges();

            TempData["SuccessMessage"] = "Book updated successfully.";
            return RedirectToAction("AllBooks");
        }
        [HttpGet]
        public ActionResult DeleteBook(int id)
        {
            var book = db.Books.FirstOrDefault(u => u.ID == id);
            if (book == null)
                return HttpNotFound();

            db.Books.Remove(book);
            db.SaveChanges();
            return RedirectToAction("AllBooks");
        }

        [HttpGet]
        public ActionResult AddBook()
        {
            return View();
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult AddBook(Book model, HttpPostedFileBase CoverImage)
        {
            if (!ModelState.IsValid)
            {
                return View(model);
            }
            string title = model.Title.Trim().ToLower();
            string author = model.Author.Trim().ToLower();

            bool exists = db.Books.Any(b =>
                b.Title.ToLower() == title &&
                b.Author.ToLower() == author);

            if (exists)
            {
                ModelState.AddModelError("", "This book already exists.");
                return View(model);
            }

            // Step 1: Save book without image URL first
            var book = new Book
            {
                Title = model.Title,
                Author = model.Author,
                Category = model.Category,
                Quantity = model.Quantity,
                Description = model.Description,
                CoverImageUrl = "" // Temporary
            };

            db.Books.Add(book);
            db.SaveChanges(); // to get Book ID

            // Step 2: Handle image
            if (CoverImage != null && CoverImage.ContentLength > 0)
            {
                string extension = Path.GetExtension(CoverImage.FileName);
                string fileName = book.ID + extension;
                string folderPath = Server.MapPath("~/BookImg/");
                string savePath = Path.Combine(folderPath, fileName);

                // Ensure directory exists
                if (!Directory.Exists(folderPath))
                {
                    Directory.CreateDirectory(folderPath);
                }

                // Save file
                CoverImage.SaveAs(savePath);

                // Update CoverImageUrl
                book.CoverImageUrl = "~/BookImg/" + fileName;
                db.SaveChanges(); // save updated path
            }

            TempData["SuccessMessage"] = "Book added successfully.";
            return RedirectToAction("AllBooks");

        }




        [HttpGet]
        public ActionResult BorrowHistory(int memberId)
        {
            var history = db.BorrowHistories
                .Where(b => b.MemberId == memberId)
                .Include("Book")
                .Include("User")
                .OrderByDescending(b => b.BorrowedAt)
                .ToList();

            var member = db.Users.Find(memberId);
            ViewBag.MemberName = member?.Name ?? "Unknown Member";

            return View(history);
        }

        [HttpGet]
        public ActionResult BorrowNotifications()
        {
            var requests = db.BorrowRequests
                .OrderByDescending(r => r.RequestedAt)
                .Include("User")
                .Include("Book")
                .ToList();

            return View(requests);
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult ApproveBorrow(int id)
        {
            var request = db.BorrowRequests.Find(id);
            if (request == null)
                return HttpNotFound();

            var book = db.Books.Find(request.BookId);
            if (book == null)
            {
                TempData["ErrorMessage"] = "Book not found.";
                return RedirectToAction("BorrowNotifications");
            }

            string mailAddress = request.User.Email;
            string bookName = request.Book.Title;
            string message = "";
            var returnSMS = "";


            if (book.Quantity < 1)
            {

                message = $"<p>Dear {request.User.Name},</p>" +
                                 $"<p>Thank you for your request for the book <strong>\"{bookName}\"</strong>.</p>" +
                                 $"<p>Unfortunately, the book is currently <span style='color:orange;'><strong>unavailable</strong></span> due to all copies being issued.</p>" +
                                 "<p>We kindly ask for your patience, as the book will be available again once it is returned by other members.</p>" +
                                 "<p>We will notify you as soon as it becomes available for request.</p>" +
                                 "<br><p>Thank you for your understanding and continued interest in our library.</p>" +
                                 "<p>Best regards,<br/>Library Management Team</p>";


                returnSMS  = sendmail(mailAddress, message, "Your Requested Book Is Currently Unavailable");

                TempData["ErrorMessage"] = $"Book quantity not available. Cannot approve request and {returnSMS}";
                return RedirectToAction("BorrowNotifications");
            }

            // Decrease book quantity
            book.Quantity -= 1;

            // Approve request
            request.Status = true;
            request.ApprovedAt = DateTime.Now;

            // Add to BorrowHistory
            var history = new BorrowHistory
            {
                MemberId = request.MemberId,
                BookId = request.BookId,
                BorrowedAt = DateTime.Now,
                ReturnedAt = null
            };

            db.BorrowHistories.Add(history);

            db.SaveChanges();

            message = $"<p>Dear {request.User.Name},</p>" +
                             $"<p>We are pleased to inform you that your request for the book <strong>\"{bookName}\"</strong> has been <span style='color:green;'><strong>approved</strong></span>.</p>" +
                             "<p>You may now proceed to collect the book from the library or access it as per the borrowing guidelines.</p>" +
                             "<br><p>Thank you for using our library system!</p>" +
                             "<p>Best regards,<br/>Library Management Team</p>";

            returnSMS= sendmail(mailAddress, message, "Your Book Request Has Been Approved!");
            TempData["SuccessMessage"] = $"Borrow request approved, book quantity updated, borrow history added, and {returnSMS}.";
            return RedirectToAction("BorrowNotifications");
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
                return  $"Error sending email: {ex.Message}";
            }
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult RejectBorrow(int id)
        {
            var request = db.BorrowRequests.Find(id);
            if (request == null)
                return HttpNotFound();

            request.Status = false;
            request.ApprovedAt = DateTime.Now;
            db.SaveChanges();

            string mailAddress = request.User.Email;
            string bookName = request.Book.Title;
            string message = $"<p>Dear {request.User.Name},</p>" +
                             $"<p>Thank you for your request for the book <strong>\"{bookName}\"</strong>.</p>" +
                             $"<p>We regret to inform you that your request has been <span style='color:red;'><strong>declined</strong></span> at this time.</p>" +
                             "<p>This may be due to borrowing policy restrictions, member status, or other eligibility criteria.</p>" +
                             "<p>We encourage you to review the library policies or reach out to us for further clarification.</p>" +
                             "<br><p>We appreciate your interest in our resources and look forward to serving you in the future.</p>" +
                             "<p>Best regards,<br/>Library Management Team</p>";

            string returnSMS = sendmail(mailAddress, message, "Update on Your Book Request");

            TempData["SuccessMessage"] = $"Borrow request rejected and {returnSMS}";
            return RedirectToAction("BorrowNotifications");
        }

        [HttpGet]
        public ActionResult ReturnRequests()
        {
            var returns = db.BorrowHistories
                .Include("Book")
                .Include("User")
                .Where(h => h.ReturnRequested && h.ReturnedAt == null)
                .OrderBy(h => h.BorrowedAt)
                .ToList();

            return View(returns);
        }
        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult ApproveReturn(int id)
        {
            var history = db.BorrowHistories.FirstOrDefault(h => h.ID == id);
            if (history == null || history.ReturnedAt != null)
            {
                TempData["ErrorMessage"] = "Invalid return approval.";
                return RedirectToAction("ReturnRequests");
            }

            var book = db.Books.Find(history.BookId);
            if (book != null)
            {
                book.Quantity += 1;
            }

            history.ReturnedAt = DateTime.Now;
            history.ReturnRequested = false;

            db.SaveChanges();

            TempData["SuccessMessage"] = "Return approved and book quantity updated.";
            return RedirectToAction("ReturnRequests");
        }

    }
}