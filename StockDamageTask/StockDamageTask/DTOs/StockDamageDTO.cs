namespace StockDamageTask.DTOs
{
    public class StockDamageDTO
    {
        public int? VoucherNo { get; set; }
        public string WarehouseName { get; set; }
        public string ItemName { get; set; }
        public string ItemCode { get; set; }        
        public int BatchNo { get; set; }
        public string Currency { get; set; }        
        public decimal Quantity { get; set; }
        public decimal Rate { get; set; }
        public decimal Amount_in { get; set; }
        public decimal ExchangeRate { get; set; }
        public decimal TotalAmount { get; set; }
    }
}