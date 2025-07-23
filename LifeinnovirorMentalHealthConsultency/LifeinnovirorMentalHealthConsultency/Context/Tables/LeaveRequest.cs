using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace LifeinnovirorMentalHealthConsultency.Context.Tables
{
    public class LeaveRequest
    {
        [Key]
        public int LeaveId { get; set; }

        [Required]
        public int DoctorId { get; set; }

        [Required]
        public DateTime StartDate { get; set; }

        [Required]
        public DateTime EndDate { get; set; }

        public string Reason { get; set; }

        [RegularExpression("Pending|Approved|Rejected",
            ErrorMessage = "Status must be Pending, Approved, or Rejected")]
        [StringLength(20)]
        public string Status { get; set; } = "Pending";

        public DateTime RequestedAt { get; set; } = DateTime.Now;
        public DateTime? ReviewedAt { get; set; }

        public int? AdminId { get; set; }
    }
}