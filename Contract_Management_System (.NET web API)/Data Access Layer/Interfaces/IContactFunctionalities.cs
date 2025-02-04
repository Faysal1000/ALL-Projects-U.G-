using Data_Access_Layer.Entity_Framework.Tables;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Data_Access_Layer.Interfaces
{
    public interface IContactFunctionalities
    {
        List<Contact> SearchByName(string name);
        List<Contact> SearchByEmail(string email);
        List<IGrouping<string, Contact>> GroupByCategory();
        string CsvToContact(List<Contact> contacts);
        List<Contact> BirthdayReminder();
        string DeleteDuplicateContact();
    }
}
