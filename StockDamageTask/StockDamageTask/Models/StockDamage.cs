using System;

namespace StockDamageTask.Context
{
    public partial class StockDamage
    {
        public int? VoucherNo { get; set; }
        public string GodownNo { get; set; }
        public string ItemName { get; set; }
        public string ItemCode { get; set; }
        public int BatchNo { get; set; }
        public string CurrencyName { get; set; }
        public decimal ConversionRate { get; set; }
        public int StockQuantity { get; set; }
        public int Quantity { get; set; }
        public decimal Rate { get; set; }
        public decimal Amount_in { get; set; }
        public decimal TotalAmount { get; set; }

        public string DrACHead { get; set; } = "Stock Damage";
        public int EmployeeId { get; set; }
        public System.DateTime CreatedDate { get; set; } = DateTime.Now;
        public string Comments { get; set; } 
    }
}
