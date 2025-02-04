using System;
using System.Collections.Generic;
using System.Data.Entity;
using System.Linq;
using System.Web;
using VisaProcessing.Entity_Framework.Table;

namespace VisaProcessing.Entity_Framework
{
    public class VisaProcessingContext:DbContext
    {
        public DbSet<TokenCounter> TokenCounters { get; set; }
        public DbSet<Tokens> Tokens { get; set; }
    }
}