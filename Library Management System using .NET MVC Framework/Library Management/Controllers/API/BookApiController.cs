using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Web.Http;
using Library_Management.Models;

namespace Library_Management.Controllers
{
    public class BookApiController : ApiController
    {
        [HttpGet]
        [Route("api/books/available")]
        public HttpResponseMessage GetAvailableBooks()
        {
            using (var db = new LibraryDBEntities1())
            {
                var books = db.Books
                              .Where(b => b.Quantity >= 0)
                              .Select(b => new
                              {
                                  b.Title,
                                  b.Author,
                                  b.Description,
                                  b.Quantity,
                                  b.Category

                              })
                              .ToList();

                return Request.CreateResponse(HttpStatusCode.OK, books);
            }
        }
    }
}
