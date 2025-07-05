using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using System.Web;
using System.Web.Mvc;
using Library_Management.Models;

namespace Library_Management.Controllers
{
    public class BookController : Controller
    {
        private readonly HttpClient client;

        public BookController()
        {
            client = new HttpClient
            {
                BaseAddress = new Uri("https://localhost:44303/api/books/available")
            };
        }
        // GET: Book
        public async Task<JsonResult> All()
        {
            var response = await client.GetAsync(client.BaseAddress);

            if (response.IsSuccessStatusCode)
            {
                var prescriptions = await response.Content.ReadAsAsync<IEnumerable<Book>>();
                return Json(prescriptions, JsonRequestBehavior.AllowGet);
            }

            return Json(new { error = "Failed to fetch book info" }, JsonRequestBehavior.AllowGet);
        }
    }
}