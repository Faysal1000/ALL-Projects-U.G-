using StockDamageTask.Context;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace StockDamageTask.Models
{
    public class SaveStockDamageRequest
    {
        public List<StockDamage> addedOrEditedStockDamages { get; set; }
        public List<string> deletedVoucherNos { get; set; }
    }
}