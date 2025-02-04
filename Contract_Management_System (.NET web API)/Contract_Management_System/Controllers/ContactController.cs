using Business_Logic_Layer.DTOs;
using Business_Logic_Layer.Services;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Web.Http;
using System.IO;
using System.Runtime.Remoting.Messaging;
using Contract_Management_System.Authentication;

namespace Contract_Management_System.Controllers
{
    [Logged]
    public class ContactController : ApiController
    {
        [HttpGet]
        [Route("api/contact/getall")]
        public HttpResponseMessage Get()
        {
            var data = ContactService.Get();
            return Request.CreateResponse(HttpStatusCode.OK, data);
        }
        [HttpGet]
        [Route("api/contact/get")]
        public HttpResponseMessage Get(int id)
        {
            var data = ContactService.Get(id);
            return Request.CreateResponse(HttpStatusCode.OK, data);
        }
        [HttpPost]
        [Route("api/contact/create")]
        public HttpResponseMessage Create(ContactDTO obj)
        {
            var data = ContactService.Create(obj);
            return Request.CreateResponse(HttpStatusCode.OK, data);
        }
        [HttpGet]
        [Route("api/contact/delete")]
        public HttpResponseMessage Delete(int id)
        {
            var data = ContactService.Delete(id);
            return Request.CreateResponse(HttpStatusCode.OK, data);
        }
        [HttpPost]
        [Route("api/contact/update")]
        public HttpResponseMessage Update(ContactDTO obj)
        {
            var data = ContactService.Update(obj);
            return Request.CreateResponse(HttpStatusCode.OK, data);
        }
        [HttpGet]
        [Route("api/contact/searchbyname")]
        public HttpResponseMessage SearchByName(string name)
        {
            var data = ContactService.SearchByName(name);
            return Request.CreateResponse(HttpStatusCode.OK, data);
        }
        [HttpGet]
        [Route("api/contact/searchbyemail")]
        public HttpResponseMessage SearchByEmail(string email)
        {
            var data = ContactService.SearchByEmail(email);
            return Request.CreateResponse(HttpStatusCode.OK, data);
        }
        [HttpGet]
        [Route("api/contact/groupbycategory")]
        public HttpResponseMessage GroupByCategory()
        {
            var data = ContactService.GroupByCategory();
            return Request.CreateResponse(HttpStatusCode.OK, data);
        }
        [HttpGet]
        [Route("api/contact/exporttocsv")]
        public HttpResponseMessage ExportAllContactToCSV()
        {
            var csvContent = ContactService.ExportAllContactToCSV();
            var fileBytes = Encoding.UTF8.GetBytes(csvContent);
            var response = new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new ByteArrayContent(fileBytes)
            };
            response.Content.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("application/octet-stream");
            response.Content.Headers.ContentDisposition = new System.Net.Http.Headers.ContentDispositionHeaderValue("attachment")
            {
                FileName = "Contact.csv"
            };
            return response;
        }

        [HttpPost]
        [Route("api/contact/uploadcsv")]
        public HttpResponseMessage UploadCSV()
        {
            if (!Request.Content.IsMimeMultipartContent())
            {
                return Request.CreateErrorResponse(HttpStatusCode.BadRequest, "No file uploaded.");
            }
            var provider = new MultipartMemoryStreamProvider();
            Request.Content.ReadAsMultipartAsync(provider).Wait();

            var file = provider.Contents[0];
            var fileBytes = file.ReadAsByteArrayAsync().Result;
            var csvContent = Encoding.UTF8.GetString(fileBytes);

            var result = ContactService.CsvToContact(csvContent);
            return Request.CreateErrorResponse(HttpStatusCode.OK, result);
        }
        [HttpGet]
        [Route("api/contact/birthdayreminder")]
        public HttpResponseMessage BirthdayReminder()
        {
            //var data = ContactService.BirthdayReminder();
            var data = ContactService.SendEmailForBirthdayReminder();
            return Request.CreateResponse(HttpStatusCode.OK, data);
        }
        [HttpGet]
        [Route("api/contact/deleteduplicate")]
        public HttpResponseMessage DeleteDuplicateContact()
        {
            var data = ContactService.DeleteDuplicateContact();
            return Request.CreateResponse(HttpStatusCode.OK, data);
        }
    }
}
