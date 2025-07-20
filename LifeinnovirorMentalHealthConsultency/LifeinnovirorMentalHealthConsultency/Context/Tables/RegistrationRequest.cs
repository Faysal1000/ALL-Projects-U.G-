using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace LifeinnovirorMentalHealthConsultency.Context.Tables
{
    public class RegistrationRequest
    {
        [Key]
        public int RequestId { get; set; }

        [Required]
        public int DoctorId { get; set; }
        public Doctor Doctor { get; set; }

        [Required]
        [RegularExpression("Pending|Approved|Rejected|Interview",
            ErrorMessage = "Status must be Pending, Approved, Rejected, or Interview")]
        [StringLength(20)]
        public string Status { get; set; } = "Pending";

        [Required]
        public DateTime SubmittedAt { get; set; } = DateTime.Now;

        public DateTime? ReviewedAt { get; set; }

        public int? AdminId { get; set; }
        public Admin Admin { get; set; }

        public string Notes { get; set; }
    }

}