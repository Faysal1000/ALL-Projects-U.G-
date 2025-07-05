
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using Library_Management.Models;

namespace Library_Management.Classes
{
    public class MemberBorrowOverview
    {
        public List<BorrowHistory> BorrowedBooks { get; set; }
        public List<BorrowRequest> BorrowRequests { get; set; }
    }

}