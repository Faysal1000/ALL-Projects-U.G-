using AutoMapper;
using Business_Logic_Layer.DTOs;
using Data_Access_Layer.Data_Manager;
using Data_Access_Layer.Entity_Framework.Tables;
using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Dynamic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using System.Net.Mail;
using System.Net;
using System.Runtime.Remoting.Lifetime;

namespace Business_Logic_Layer.Services
{
    public class ContactService
    {
        public object WebApiConfig { get; private set; }

        public static Mapper GetMapper()
        {
            var config = new MapperConfiguration(cfg =>
            {
                cfg.CreateMap<Contact, ContactDTO>();
                cfg.CreateMap<ContactDTO, Contact>();
            });
            var mapper = new Mapper(config);
            return mapper;
        }
        public static ContactDTO Get(int id)
        {
            return GetMapper().Map<ContactDTO>(DataManager.ContactData().Get(id));
        }
        public static List<ContactDTO> Get()
        {
            return GetMapper().Map<List<ContactDTO>>(DataManager.ContactData().Get());
        }
        public static bool Create(ContactDTO obj)
        {
            return DataManager.ContactData().Create(GetMapper().Map<Contact>(obj));
        }
        public static bool Delete(int id) 
        {
            return DataManager.ContactData().Delete(id);
        }
        public static bool Update(ContactDTO obj) 
        { 
            return DataManager.ContactData().Update(GetMapper().Map<Contact>(obj));
        }
        public static List<ContactDTO> SearchByName(string name)
        {
            return GetMapper().Map<List<ContactDTO>>(DataManager.ContactFunction().SearchByName(name));
        }

        public static List<ContactDTO> SearchByEmail(string email)
        {
            return GetMapper().Map<List<ContactDTO>>(DataManager.ContactFunction().SearchByEmail(email));
        }

        public static List<IGrouping<string, ContactDTO>> GroupByCategory()
        {
            var groupedData = DataManager.ContactFunction().GroupByCategory();
            var data = (from Data in groupedData
                        from contact in Data
                        let contactDTO = GetMapper().Map<ContactDTO>(contact)
                        group contactDTO by contactDTO.Relation into newGroup
                        select newGroup).ToList();
            return data;
        }
        public static string ExportAllContactToCSV()
        {
            var contacts = GetMapper().Map<List<ContactDTO>>(DataManager.ContactData().Get());
            var csv = new StringBuilder();
            csv.AppendLine("Id,Name,Phone,Email,Address,Note,Birthday,Relation,SocialMedia");
            foreach (var contact in contacts)
            {
                var addressFormatted = $"\"{contact.Address}\"";
                csv.AppendLine($"{contact.Id},{contact.Name},{contact.Phone},{contact.Email},{addressFormatted},{contact.Note},{contact.Birthday},{contact.Relation},{contact.SocialMedia}");
            }
            return csv.ToString();
        }
        public static string CsvToContact(string csvContent)
        {
            var rows = csvContent.Split(new[] { "\r\n", "\n" }, StringSplitOptions.RemoveEmptyEntries);

            var contacts = new List<ContactDTO>();
            try
            {
                for (int i = 1; i < rows.Length; i++)
                {
                    var columns = ParseCsvRow(rows[i]);

                    var contact = new ContactDTO
                    {
                        Name = columns[1],
                        Phone = columns[2] ?? string.Empty,
                        Email = columns[3] ?? string.Empty,
                        Address = columns[4] ?? string.Empty,
                        Note = columns[5] ?? string.Empty,
                        Birthday = string.IsNullOrEmpty(columns[6]) ? (DateTime?)null : DateTime.Parse(columns[6]),
                        Relation = columns[7] ?? string.Empty,
                        SocialMedia = columns[8] ?? string.Empty
                    };

                    contacts.Add(contact);
                }
            } catch(Exception e) {
                return "Error! CSV file is not in correct format! Exception: "+ e.Message;
            }
            return DataManager.ContactFunction().CsvToContact(GetMapper().Map<List<Contact>>(contacts));
        }

        // This method uses regular expressions to correctly handle quoted fields with commas
        private static string[] ParseCsvRow(string row)
        {
            var regex = new System.Text.RegularExpressions.Regex(
                @"(?<=^|,)(\"".*?\""|[^,]*)",
                System.Text.RegularExpressions.RegexOptions.IgnoreCase);

            var matches = regex.Matches(row);
            var values = matches.Cast<System.Text.RegularExpressions.Match>()
                                .Select(m => m.Value.Trim('"'))
                                .ToArray();
            return values;
        }
        public static string BirthdayReminder()
        {
            var data = GetMapper().Map<List<ContactDTO>>(DataManager.ContactFunction().BirthdayReminder());
            if (data.Count == 0)
            {
                return "No Birthday Today!";
            }

            string message = "";
            foreach (var contact in data)
            {
                var age = DateTime.Today.Year - contact.Birthday.Value.Year;
                message += $"{contact.Name}'s ({age}th), ";
            }
            return "Today is "+message+ " birthday! Please wish them a happy birthday!";
        }
        public static string SendEmailForBirthdayReminder()
        {
            var message = BirthdayReminder();
            if (message == "No Birthday Today!") { 
                return "No one from you contact list has a birthday today!";
            }
            var smtpClient = new SmtpClient("smtp.gmail.com")
            {
                Port = 587,
                Credentials = new NetworkCredential("faysalahmmed4200@gmail.com", "ncxaxuhsnaqrutnr"), 
                EnableSsl = true,
            };
            var mailMessage = new MailMessage
            {
                From = new MailAddress("faysalahmmed4200@gmail.com"), 
                Subject = "Birthday Reminder",
                Body = message,
                IsBodyHtml = true
            };
            mailMessage.To.Add("faysalahmmed4200@gmail.com");
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
        public static string DeleteDuplicateContact()
        {
            return DataManager.ContactFunction().DeleteDuplicateContact();
        }
    }
}
