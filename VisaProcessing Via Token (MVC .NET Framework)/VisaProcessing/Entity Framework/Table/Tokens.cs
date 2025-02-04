using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace VisaProcessing.Entity_Framework.Table
{
    public class Tokens
    {
        public int Id { get; set; }
        public string TokenNo { get; set; }
        public bool Service { get; set; }
    }
}