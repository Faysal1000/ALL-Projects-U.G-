using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace LifeinnovirorMentalHealthConsultency.Models
{
    public class DoctorStatusUpdateRequestModel
    {
        [Required]
        public int DoctorId { get; set; }
        [RegularExpression("Pending|Approved|Interview|Rejected",
            ErrorMessage = "Status must be Pending, Interview, Approved, or Rejected")]
        public string NewStatus { get; set; }

    }
}