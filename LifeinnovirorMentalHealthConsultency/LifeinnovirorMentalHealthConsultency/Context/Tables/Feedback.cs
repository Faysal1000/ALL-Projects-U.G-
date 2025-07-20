using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace LifeinnovirorMentalHealthConsultency.Context.Tables
{
    public class Feedback
    {
        [Key]
        public int FeedbackId { get; set; }

        public int AppointmentId { get; set; }

        public int PatientId { get; set; }

        [Range(1, 5)]
        public int Rating { get; set; }

        public string Comments { get; set; }

        public DateTime SubmittedAt { get; set; } = DateTime.Now;
    }
}