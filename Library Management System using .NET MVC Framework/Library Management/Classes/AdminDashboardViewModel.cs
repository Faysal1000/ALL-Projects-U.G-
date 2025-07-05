using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace Library_Management.Classes
{
    public class AdminDashboardViewModel
    {
        public int TotalBorrowed { get; set; }
        public int TotalInStock { get; set; }

        public List<DailyStat> BorrowedLast7Days { get; set; }
        public List<DailyStat> ReturnedLast7Days { get; set; }

    }

}