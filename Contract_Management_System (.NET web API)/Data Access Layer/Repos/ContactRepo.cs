using Data_Access_Layer.Entity_Framework.Tables;
using Data_Access_Layer.Interfaces;
using System;
using System.Collections.Generic;
using System.Data.Entity;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace Data_Access_Layer.Repos
{
    internal class ContactRepo : Repo, IRepo<Contact, int, bool>, IContactFunctionalities
    {
        public bool Create(Contact obj)
        {
            db.Contacts.Add(obj);
            return db.SaveChanges()>0;
        }
        public Contact Get(int id)
        {
            return db.Contacts.Find(id);
        }

        public List<Contact> Get()
        {
            return db.Contacts.ToList();
        }
        public bool Delete(int id)
        {
            var exob = Get(id);
            db.Contacts.Remove(exob);
            return db.SaveChanges() > 0;
        }
        public bool Update(Contact obj)
        {
            var exob = Get(obj.Id);
            db.Entry(exob).CurrentValues.SetValues(obj);
            return db.SaveChanges()>0;
        }

        public List<Contact> SearchByName(string name)
        {
            var data = (from Data in db.Contacts
                        where Data.Name == name
                        select Data).ToList();
            return data;
        }

        public List<Contact> SearchByEmail(string email)
        {
            var data = (from Data in db.Contacts
                        where Data.Email == email
                        select Data).ToList();
            return data;
        }

        public List<IGrouping<string, Contact>> GroupByCategory()
        {
            var data = (from contact in db.Contacts
                        group contact by contact.Relation).ToList();
            return data;
        }

        public string CsvToContact(List<Contact> contacts)
        {
            int unique = 0, duplicate = 0;
            foreach (var contact in contacts)
            {   
                var exobj = db.Contacts.FirstOrDefault(c => c.Phone == contact.Phone); 
                if (exobj == null)
                {
                    db.Contacts.Add(contact);
                    unique++;
                }
                else
                {
                    contact.Id = exobj.Id;
                    db.Entry(exobj).CurrentValues.SetValues(contact);
                    duplicate++;
                }
            }
            if (db.SaveChanges() > 0)
            {
                return unique + " unique contact inserted and " + duplicate + " contact undated";
            }
            else
            {
                return "Something went wrong while processing csv file.";
            }
        }

        public List<Contact> BirthdayReminder()
        {
            var data = (from contact in db.Contacts
                        where contact.Birthday.HasValue &&
                        contact.Birthday.Value.Month == DateTime.Today.Month &&
                        contact.Birthday.Value.Day == DateTime.Today.Day
                        select contact).ToList();
            return data;
        }

        public string DeleteDuplicateContact()
        {
            var duplicates = db.Contacts
                .GroupBy(contact => contact.Phone)
                .Where(group => group.Count() > 1)
                .SelectMany(group => group.OrderBy(c => c.Id).Skip(1)) 
                .ToList();

            int removedCount = duplicates.Count;
            db.Contacts.RemoveRange(duplicates);
            db.SaveChanges();

            return $"{removedCount} duplicate contact(s) were removed.";
        }
    }
}
