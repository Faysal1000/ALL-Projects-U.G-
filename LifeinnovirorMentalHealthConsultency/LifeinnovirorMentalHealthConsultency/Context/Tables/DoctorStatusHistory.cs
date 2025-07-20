using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace LifeinnovirorMentalHealthConsultency.Context.Tables
{
    public class DoctorStatusHistory
    {
        [Key]
        public int StatusId { get; set; }

        [Required]
        public int DoctorId { get; set; }

        [Required]
        [RegularExpression("Available|Unavailable|Consultation",
            ErrorMessage = "Status must be Available, Unavailable, or Consultation")]
        [StringLength(20)]
        public string Status { get; set; } 

        public DateTime ChangedAt { get; set; } = DateTime.Now;
    }

}