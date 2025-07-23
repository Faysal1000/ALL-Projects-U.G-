using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace LifeinnovirorMentalHealthConsultency.Context.Tables
{
    public class Payment
    {
        [Key]
        public int PaymentId { get; set; }

        public int AppointmentId { get; set; }

        [Required]
        public decimal Amount { get; set; }

        public DateTime PaidAt { get; set; } = DateTime.Now;

        public string Method { get; set; }

        public string TransactionId { get; set; }

        [RegularExpression("Pending|Completed|Failed",
            ErrorMessage = "Status must be Pending, Completed, or Failed")]
        [StringLength(20)]
        public string Status { get; set; } = "Pending";
    }
}