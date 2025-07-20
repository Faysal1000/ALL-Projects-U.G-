using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace LifeinnovirorMentalHealthConsultency.Context.Tables
{
    public class Report
    {
        [Key]
        public int ReportId { get; set; }

        public int AppointmentId { get; set; }

        public int DoctorId { get; set; }

        [Required]
        public string UploadUrl { get; set; }

        public DateTime UploadedAt { get; set; } = DateTime.Now;
    }
}