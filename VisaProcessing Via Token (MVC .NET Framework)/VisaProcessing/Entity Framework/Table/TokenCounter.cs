using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace VisaProcessing.Entity_Framework.Table
{
    public class TokenCounter
    {
        public int Id { get; set; }
        public int MedCounter { get; set; }
        public int TrCounter { get; set; }
        public int BsCounter { get; set; }
        public int GoCounter { get; set; }
        public int TotalCounter { get; set; }
        public DateTime date { get; set; }
    }
}