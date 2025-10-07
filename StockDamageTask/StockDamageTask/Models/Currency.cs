namespace StockDamageTask.Models
{    
    public partial class Currency
    {
        public int CurrencyId { get; set; }
        public string CurrencyName { get; set; }
        public decimal ConversionRate { get; set; }
    }
}
